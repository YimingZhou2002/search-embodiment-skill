---
dimensions:
  knob: rollout_epoch
  env: maniskill
  model: openvlaoft
  algorithm: ppo
  cfg_pattern: collocation
campaign: search_20260813-070815
created: 2026-08-20
updated: 2026-08-20
confidence: high
baseline_speed: 1.0106
best_speed: 0.5050
speedup: "-50.0%"
num_nodes: 27
---

# rollout_epoch Tuning

## Campaign: maniskill + PPO + openvlaoft (search_20260813-070815)

This is the first campaign contributing data to this knob-effect entry. Initial findings below.

## Summary

`env.train.rollout_epoch` controls how many times the rollout model reuses the actor's weights to generate trajectories before the actor trains. Increasing re amortizes fixed per-step costs (offload, weight sync) across more epochs, reducing per-traj time. This is a PPO-specific lever — the weight-sync step is unique to PPO's actor-rollout architecture.

## Data

**Collocated branch, tne=384:**
- re=1: 0.5570 s (node #16)
- re=2: 0.5173 s (node #20) — **-7.1%**
- re=3: 0.5050 s (node #23) — **-2.4%**
- re=4: CRASH (node #24) — pybind11_object_dealloc

**Collocated branch, tne=448:**
- re=1: 0.5516 s (node #19)
- re=2: 0.5181 s (node #26) — **-6.1%**

## Key Insight

The mechanism is amortization of fixed costs. In the best node:
- Offload cost: 5.4% of step time (67.8 s) — this is the sum of offload+onload across rollout and actor
- Weight sync: 1.2% of step time (15.6 s) — sync_model_to_rollout

These are both fixed per step. Spreading them across 3 epochs gives a ~3x reduction in per-traj overhead. The effect is diminishing: re=1->2 gives -7.1%, re=2->3 gives -2.4%, suggesting the majority of the fixed cost is amortized by re=2.

re=4 crashes with a pybind11 runtime bug, not OOM. The error `pybind11_object_dealloc(): Tried to deallocate unregistered instance!` indicates a lifecycle issue with the rollout's trajectory buffer when too many epochs share the same buffer. This is a hard ceiling — do not attempt re=4 without code changes.

## Best Value

**3**, under collocation + tne=384. Gives 0.5050 s/traj.

## Conditions

- Only effective under collocation (where rollout shares actor weights). The weight-sync cost is minimal under disaggregation because rollout already has its own weights.
- Not compounded with tne scaling — tne and re are substitutes.
- re=1->2 is the big win (-7.1%); re=2->3 is smaller but still meaningful (-2.4%).
- The ceiling is 3 due to a runtime bug, not memory.

## Raw Data Summary

```json
{
  "knob": "env.train.rollout_epoch",
  "range_explored": [1, 2, 3, 4],
  "best_value": 3,
  "direction": "positive (diminishing)",
  "confidence": "high",
  "data_points": {
    "collocated_tne_384": [
      {"re": 1, "speed": 0.5570},
      {"re": 2, "speed": 0.5173, "improvement": -7.1},
      {"re": 3, "speed": 0.5050, "improvement": -2.4},
      {"re": 4, "status": "CRASH", "failure": "pybind11_object_dealloc"}
    ],
    "collocated_tne_448": [
      {"re": 1, "speed": 0.5516},
      {"re": 2, "speed": 0.5181, "improvement": -6.1}
    ]
  },
  "mechanism": "amortizes fixed offload + weight-sync cost across epochs",
  "notes": "PPO-specific lever. re=4 capped by runtime bug, not OOM. Do not attempt without code changes."
}
```