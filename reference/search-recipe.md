# Search recipe — beam search over configs

How the search proposes, evaluates, and prunes config variants. The deterministic
bookkeeping is in `helpers/search_store.py` (node store + tree) and
`helpers/preflight.py` (validity); **the proposal step is yours** — read the
frontier node's diagnosis and pick knob deltas. Objective = `step_time_per_traj_s`
(from `diagnosis.json`), **lower is better**.

## The loop (beam width 2, branching 2)

```
cold start:  run baseline → diagnose → node #0 (root)
for round r in 1..10:
    frontier = 2 best OK nodes by objective        # search_store.py frontier --k 2 --max-children 2
    for each frontier node f:
        read f's diagnosis.json + REPORT.md
        propose 2 knob deltas that should lower the objective   # you, using the playbook below
    for each of the (up to) 4 proposals d on parent f:
        dedup d       → if duplicate, skip (reuse known objective)
        preflight d   → if invalid, record FAILED/CONFIG_INVALID, re-propose or drop
        else: launch trial (short) → diagnose → add node, set objective
    print tree + leaderboard
finish: best node → SEARCH_REPORT.md
```

4 trials/round are **sequential** (each needs all GPUs). Budget accordingly.

## Proposing deltas from a diagnosis

Map the frontier node's dominant signal (from its `diagnosis.json`, matched via
`diagnosis-playbook.md`) to 1–2 knob changes in `knob-schema.md`. Priority
mapping:

| Diagnosis signal | Propose |
|---|---|
| Generation-bound (`generation_pct`≫`training_pct`, high `actor_idle`) | ↑ `total_num_envs`; consider disaggregating placement so training overlaps generation |
| Too-few-envs / poor amortization (low tne, high per-traj) | ↑ `total_num_envs` (next legal multiple of 8) |
| CPU-bound env (`cpu_saturation` env multicore, env% of gen high) | ↑ `total_num_envs`; `env.train.enable_offload=false` if headroom |
| Offload tax + safe memory (`offload_cost_pct` high, `oom_risk=safe`) | `env.train.enable_offload=false` (never rollout under colocation) |
| Memory headroom (`max_used_pct` low) | ↑ `micro_batch_size` (stay in {…40,80}, under OOM knee) |
| OOM / `oom_risk=high` | ↓ `micro_batch_size`; ensure `rollout.enable_offload=true`; `gradient_checkpointing=true` |
| Rollout inference-bound (`predict%` of gen high) | ↑ `rollout` placement width; tune `pipeline_stage_num` |

**Strategy over rounds** (mirrors the DAG-search wisdom):
- **Early rounds → breadth.** Off the same parent, probe *different* single knobs
  (one delta each) to build a sensitivity map. Prefer single-knob deltas — they're
  attributable.
- **Later rounds → depth.** Once the leaderboard shows which knob moves the
  objective most, keep expanding the best node, refining that knob (and stacking a
  second compatible one).
- **One knob per delta** unless you have evidence two are independent. Two
  proposals per node should differ (don't waste a branch on a near-duplicate).
- **Respect memory.** Don't propose past the OOM knee (mbs≈108) or
  `rollout.enable_offload=false` unless placement was disaggregated first — those
  are known OOMs, wasted trials.

## Guardrails (always, before launching)

1. **dedup** (`search_store.py dedup --parent f --overrides d`): if it returns a
   duplicate, skip the run and reuse that node's objective — don't burn GPU on a
   config already tried.
2. **preflight** (`preflight.py --overrides <resolved-or-delta>`): if invalid,
   record a FAILED node (`--status FAILED --failure CONFIG_INVALID`) *without*
   running, and propose a different delta.
3. Overrides are **cumulative**: a child's delta is applied on top of its parent's
   resolved config. Propose the *incremental* change only; the store resolves the
   full config and computes the dedup SHA.

## Stopping early

If a full round produces no OK node that beats the current best (all duplicates,
invalid, OOM, or slower), you may stop before round 10 and report the best so far —
note the plateau in `SEARCH_REPORT.md`.
