# Search recipe — beam search over configs

How the search proposes, evaluates, and prunes config variants. The deterministic
bookkeeping is in `helpers/search_store.py` (node store + tree) and
`helpers/preflight.py` (validity); **the proposal step is yours** — read the
frontier node's diagnosis and pick knob deltas. Objective = `step_time_per_traj_s`
(from `diagnosis.json`), **lower is better**.

## Scoring: composite objective (A4)

The frontier is selected by **composite score**, not raw `step_time_per_traj_s`.
This prevents the beam from collapsing into a local optimum (a fast but memory-tight
dead end). The composite score is:

```
composite = step_time_per_traj_s × (1 − α × max(0, headroom_gib − 5) / baseline_headroom)
            − β × sqrt(ln(rounds + 1) / (children + 1))
            + γ × max(0, baseline_success − node_success) / baseline_success
```

| Term | Coeff | What it does |
|------|-------|-------------|
| Headroom potential | α=0.15 | Gives nodes with GPU memory headroom a discount — they can still grow (raise mbs, disable offload) |
| Exploration bonus | β=0.05 | UCB-style bonus for nodes with fewer children — ensures breadth in early rounds |
| Quality penalty | γ=0.5 | Penalizes nodes that degrade `success_once` vs baseline — prevents false wins |

Tune α/β/γ via `search_store.py frontier --composite --alpha 0.15 --beta 0.05 --gamma 0.5`.
Pass `--no-composite` to revert to raw objective ranking.

## The loop (beam width 2, branching 2)

```
cold start:  run baseline → diagnose → node #0 (root)
for round r in 1..10:
    frontier = 2 best OK nodes by composite score  # search_store.py frontier --k 2 --max-children 2
    for each frontier node f:
        read f's diagnosis.json + REPORT.md
        propose 2 knob deltas that should lower the objective   # you, using the playbook below
    for each of the (up to) 4 proposals d on parent f:
        dedup d       → if duplicate, skip (reuse known objective)
        preflight d   → if invalid, record FAILED/CONFIG_INVALID, re-propose or drop
        else: launch trial (short) → diagnose+REPORT → add node, set objective
    print tree + leaderboard
    run exhaustion-check → if should_stop: break early   # B4: stop only when genuinely exhausted
finish: best node → SEARCH_REPORT.md
```

4 trials/round are **sequential** (each needs all GPUs). Budget accordingly.

## Proposing deltas from a diagnosis

Map the frontier node's dominant signal (from its `diagnosis.json`, matched via
`diagnosis-playbook.md`) to 1–2 knob changes in `knob-schema.md`. Priority
mapping:

| Diagnosis signal | Propose |
|---|---|
| Too-few-envs / poor amortization (low tne, high per-traj) | ↑ `total_num_envs` (next legal multiple of 8) |
| CPU-bound env (`cpu_saturation` env multicore, env% of gen high) | ↑ `total_num_envs`; `env.train.enable_offload=false` if headroom |
| Offload tax + safe memory (`offload_cost_pct` high, `oom_risk=safe`) | `env.train.enable_offload=false` (never rollout under colocation) |
| Memory headroom (`max_used_pct` low) | ↑ `micro_batch_size` (ensure `global_batch_size%(micro_batch_size*actor_world)`, under OOM knee) |
| OOM / `oom_risk=high` | ↓ `micro_batch_size`; ensure `rollout.enable_offload=true`; `actor.enable_offload==true` |
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

## Using the wiki (cross-campaign knowledge)

Before proposing knob deltas, query the wiki for relevant past experience:

```bash
python "$SKILL/wiki/wiki_index.py" query \
  --wiki-dir "$SKILL/wiki" \
  --env "$ENV_NAME" --model "$MODEL_NAME" --algorithm "$ALGO_NAME"
```

This returns paths to wiki entries ranked by relevance. Read the top 1-3 entries.

### How to use wiki knowledge

1. **Check known knob sensitivity.** If the wiki entry for this env/model/algo says
   a knob has a strong effect in a certain direction, prioritize that direction.
   Example: wiki says "ManiSkill + OpenVLA-OFT: collocation (env, rollout in 0-7)
   gives -50% improvement" -> propose collocation changes early.

2. **Avoid known dead ends.** If the wiki lists OOM walls or invalid configs for
   this env/model/algo, do not propose them. Example: wiki says "micro_batch_size
   > 108 -> OOM for OpenVLA-OFT on 80GiB GPUs" -> keep mbs <= 108.

3. **Get component memory footprint.** The wiki's Memory & Bottleneck Profile section
   tells you the peak memory of each component. Use this to make informed offload
   decisions: if actor+rollout+env > 80% of total GPU memory, keep offload enabled.

4. **Cross-campaign knob effects.** The `knob-effect/` entries aggregate experience
   across all campaigns. If a knob consistently shows a positive effect across
   different env/model combinations, it's worth trying. If it shows mixed results,
   note the conditions under which it works.

### Priority

- **High-confidence wiki entries** override the default playbook for their
  specific dimension. If a high-confidence wiki entry says a knob is harmful,
  do not propose it.
- **Medium/low-confidence entries** are advisory - use them as hints but still
  follow the diagnosis-playbook patterns.
- **No wiki entry** -> fall back to the full diagnosis-playbook heuristic.

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

## Stopping early (B4 — combined exhaustion check)

After each round, run the exhaustion check to decide whether to stop:

```bash
python "$SKILL/helpers/search_store.py" exhaustion-check \
  --campaign-dir "$CAMPAIGN_DIR" --k 2 --max-children 2 \
  --min-headroom-gib 5.0 --min-rounds 5
```

The search stops **only when ALL four conditions are met**:

| # | Condition | Meaning |
|---|-----------|---------|
| 1 | **Plateau** | Best objective hasn't improved for 2 consecutive rounds |
| 2 | **Headroom exhausted** | Every frontier node has < 5 GiB headroom — can't push mbs/offload further |
| 3 | **Offload trade exhausted** | No frontier node has env offload enabled with enough headroom to disable it |
| 4 | **Minimum rounds** | At least 5 rounds completed — don't stop before the search has had a fair chance |

If the verdict is `should_stop: true`, stop and report the best so far.
Otherwise, continue — the search still has viable expansion paths.
Note the verdict and reasons in `SEARCH_REPORT.md`.

This replaces the old single-round plateau rule. A single unlucky round (all
proposals hit duplicates or OOM) no longer kills the search; the search only
stops when it is genuinely exhausted.
