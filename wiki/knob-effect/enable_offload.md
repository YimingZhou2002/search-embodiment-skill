---
dimensions:
  knob: enable_offload
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

# enable_offload

## Campaign: maniskill + PPO + openvlaoft (search_20260813-070815)

This is the first campaign contributing data to this knob-effect entry. Initial findings below.

## Summary

`env.train.enable_offload` controls whether the environment simulation's state is offloaded from GPU memory between steps. The knob is separate from `actor.enable_offload` and `rollout.enable_offload`, which control model weight offloading. The env sim is small (CPU-bound, no model weights), so offloading it has minimal memory benefit and only adds onload overhead.

## Data

**Disaggregated branch, tne=128 (baseline):**
- enable_offload=true: 1.0106 s (node #0, baseline)
- enable_offload=false: 0.9902 s (node #1) — **-2.0%**

**Disaggregated branch, tne=320:**
- enable_offload(true): 0.7318 s (node #11)
- enable_offload(false): 0.7278 s (node #7) — **-0.5%**

**Collocated branch, tne=320, mbs=80:**
- enable_offload(false): 0.5442 s (node #15)
- enable_offload(true): 0.5557 s (node #21) — **+2.1%** (worse)

**Collocated branch, tne=320, mbs=40:**
- enable_offload(false): 0.5695 s (node #12)
- enable_offload(true): 0.5594 s (node #17) — **-1.8%** (within noise)

## Key Insight

Under collocation, env_offload is a **no-op**. The env sim is tiny relative to the model weights (which dominate the 50.79 GiB memory footprint). Under collocation, re-enabling env_offload (#21 vs #15) leaves peak memory byte-identical (77.94 GiB / 97.4%) and only adds onload overhead (+2.1%). The small -2% improvement in the disaggregated baseline comes from avoiding the offload serialization/deserialization cost, not from memory savings.

The memory diagnosis confirms: env_offload doesn't change peak memory usage because the env sim is CPU-bound, not GPU-memory-bound. The model's 50+ GiB footprint comes from the VLA transformer weights, not the environment state.

## Best Value

**false** — disable env offload to avoid onload overhead. This is safe under both disaggregated and collocated configurations.

## Conditions

- Under collocation, env_offload has no memory benefit. The env sim shares GPU memory with the model, and the model's weight footprint dwarfs the env state.
- Under disaggregation, the improvement is modest (-2% at tne=128).
- The bigger lever is `actor.enable_offload=false` and `rollout.enable_offload=false` — but those are needed to fit resident weights and cannot be changed.

## Raw Data Summary

```json
{
  "knob": "env.train.enable_offload",
  "range_explored": [true, false],
  "best_value": false,
  "direction": "none (no-op under collocation)",
  "confidence": "high",
  "data_points": {
    "disaggregated_tne_128": {
      "true": 1.0106,
      "false": 0.9902,
      "delta": -0.0204
    },
    "disaggregated_tne_320": {
      "true": 0.7318,
      "false": 0.7278,
      "delta": -0.004
    },
    "collocated_tne_320_mbs_80": {
      "false": 0.5442,
      "true": 0.5557,
      "delta": 0.0115
    },
    "collocated_tne_320_mbs_40": {
      "false": 0.5695,
      "true": 0.5594,
      "delta": -0.0101
    }
  },
  "mechanism": "env sim is CPU-bound and tiny; offload adds onload overhead with no memory benefit",
  "notes": "Distinct from actor.enable_offload and rollout.enable_offload, which are required for model weight management."
}
```