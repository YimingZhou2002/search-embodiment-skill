# Search Report Template — RLinf embodiment config beam search

The search report is the deliverable for a beam-search tuning campaign. Save as
`<log_dir>/SEARCH_REPORT.md`. A busy reader should get the winner (best per-trajectory
time + delta vs baseline) in 10 seconds, then drill down into the search tree,
leaderboard, and per-round progression.

Fill every `<...>` from the search run's trial logs and `diagnosis.json` outputs.
The knob deltas are cumulative overrides vs the baseline config — each node inherits
its parent's overrides plus one new knob change.

---

## Template

```markdown
# Search Report — `<config_name>` Embodiment Config Beam Search

**Campaign:** `<search_run_id>`
**Started:** `<start_ts>` UTC | **Completed:** `<end_ts>` UTC
**Method:** Beam search, width=`<W>`, branching=`<B>`, STEPS=`<N>`
**Rounds completed:** `<R>` (of planned `<R_planned>`)
**GPUs:**  `<GPU number>` × `<GPU model>` - `<per GPU memory>`
**Objective:** `step_time_per_traj_s` (lower is better)

---

## Winner: Node #`<N>` — **`<obj>` s/traj** (`<delta_pct>`% vs baseline)

## Winning Config

```yaml
# Node #3 "tne256_env_off" — 2.0702 s/traj (−35.0% vs baseline 3.1854 s/traj)
cluster:
  component_placement:
    actor: `<>`
    env: `<>`
    rollout: `<>`
env:
  train:
    total_num_envs: `<>`       
    enable_offload: `<>`
    rollout_epoch: `<>`
rollout:
  enable_offload: `<>`
  pipeline_stage_num: `<>`
actor:
  micro_batch_size: `<>`
  global_batch_size: `<>`
  enable_offload: `<>`
```

**Overrides from baseline:**
- `<delta knob>`=`<delta value>`


---

## Leaderboard

| Rank | Node | Config | Per-Traj (s) | vs Baseline |
|------|------|--------|-------------|-------------|
| **1** | **#`<N>`** | **`<summary>`** | **`<obj>`** | **`<delta>`%** |
| 2 | #`<N>` | `<summary>` | `<obj>` | `<delta>`% |
| 3 | #0 | baseline (no overrides) | `<baseline_obj>` | — |
| ... | ... | ... | ... | ... |

---

## Search Tree

```
#0 [OK] obj=<baseline> (baseline)
├─ #1 [OK] obj=<obj> {<delta_json>}
│  ├─ #3 [OK] obj=<obj> {<delta_json>}
│  │  ├─ #8 [OK] obj=<obj> {<delta_json>}
│  │  │  ├─ #N [OK] obj=<obj> {<delta_json>}  ← WINNER
│  │  │  └─ #M [OK] obj=<obj> {<delta_json>}
│  │  └─ #K (skipped — plateau)
│  └─ #4 [OK] obj=<obj> {<delta_json>}
│     └─ #9 [OK] obj=<obj> {<delta_json>}
└─ #2 [OK] obj=<obj> {<delta_json>}
   └─ (skipped — plateau)
```

---

## Per-Round Progression

### Round 0 — Cold start
- **#0 baseline:** `<baseline_obj>` s/traj

### Round 1 — `<theme: e.g. Single-knob probes>`
- **#`<N>` `<knob=value>`:** `<obj>`s (`<delta>`%) — `<one-line takeaway>`
- **#`<M>` `<knob=value>`:** `<obj>`s (`<delta>`%) — `<one-line takeaway>`

### Round 2 — `<theme>`
- **#`<N>` `<summary>`:** `<obj>`s (`<delta>`%) — `<one-line takeaway>`
- ...

### Round N — `<theme>`
- ...

---

## Knob Sensitivity Map

| Knob | Effect | Winning Value |
|------|--------|---------------|
| `<knob.path>` | `<tuned-range summary + direction + magnitude>` | **`<value>`** |
| `<knob.path>` | `<summary>` | **`<value>`** (keep default) |
| ... | ... | ... |

---

## Key Insights

1. **`<headline finding>`.** `<explanation with cited numbers>`.

2. **`<headline finding>`.** `<explanation>`.

3. **`<headline finding>`.** `<explanation>`.

4. **`<headline finding>`.** `<explanation>`.

---

## Conclusion

The search found a **`<delta_pct>`% improvement** over the baseline `<config_name>` config by:
- `<action 1>`
- `<action 2>`
- `<action 3>`

Further improvements likely require `<beyond-config-tuning: e.g. model-level optimizations,
hardware changes, algorithm changes>` rather than config tuning.
```

---

## Style rules

- **Winner first.** Open with the winning node + config so the reader gets the answer immediately.
- **Cumulative configs.** Each node's knobs are cumulative overrides from root — show the JSON
  delta in the tree, not the full config. The winner section shows both cumulative deltas and
  resolved knobs for clarity.
- **Cite numbers.** Every claim in Key Insights names a specific measurement ("saves ~0.06s/traj",
  not "saves some time").
- **Round themes.** Label each round with the idea behind the probes (e.g. "Single-knob probes",
  "Pipeline & batch size", "Actor offload breakthrough") — it tells the story of the search.
- **Sensitivity map.** One row per knob that was explored; state the direction and magnitude of
  effect, and whether the winning value differs from default.
- **Plateau / skipped nodes.** Mark skipped nodes in the tree with the reason
  (e.g. "skipped — plateau", "skipped — OOM").
- **Conclusion is actionable.** List the concrete config changes, not abstract principles.
- **Resolved knobs** include defaults that weren't changed — this is the full config the winner
  actually ran with.

## Anti-patterns

- ❌ Burying the winner — it belongs in the first section, not after the tree.
- ❌ Omitting the baseline in the leaderboard — the reader needs the reference point.
- ❌ Qualitative knob descriptions ("makes things faster") without direction and magnitude.
- ❌ Skipping the per-round progression — it captures the search narrative and why certain
  branches were abandoned.
- ❌ Presenting deltas as absolute configs in the tree — use the JSON override dict.
- ❌ No knob sensitivity map — readers scanning for "what matters" should get it in one table.
- ❌ Generic conclusion ("config tuning helped") — list the specific changes and their
  individual contributions.
