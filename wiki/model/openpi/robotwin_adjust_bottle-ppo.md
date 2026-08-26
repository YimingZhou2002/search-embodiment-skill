---
dimensions:
  env: robotwin_adjust_bottle
  model: openpi
  algorithm: ppo
  cfg_pattern: rollout_offload_disabled
campaign: search_robotwin-ppo-openpi-20260814-071647
created: 2026-08-20
updated: 2026-08-20
confidence: high
baseline_speed: 1.3031
best_speed: 1.1826
speedup: "-9.2%"
num_nodes: 7
---

# openpi Model: robotwin_adjust_bottle + PPO

## Key Findings

1. **openpi is a small model (~3B params, ~7 GB) -- rollout offload is pure churn.**
   The atomic sweep measured `onload peak_alloc~7.0 GB` for the openpi rollout model. 
   This is in stark contrast to openvla (7B, ~25 GB) where offload is essential. 
   Keeping the openpi rollout model resident on GPU saves 9.2% wall time with no OOM risk 
   (peak memory 49.3 GiB out of 80 GiB).

2. **micro_batch_size=32 is the sweet spot; 64 creates memory contention.**
   Raising mbs from 32 to 64 on the winner (rollout_off already enabled) regressed performance 
   by 7.9% (1.1826 -> 1.2765 s/traj). Training time decreased by 3.9 s but generation time 
   increased by 7.5 s -- the extra memory used by larger micro-batches slowed rollout predict 
   via GPU memory contention.

3. **Actor offload is beneficial for openpi despite the small model size.**
   Disabling actor offload (node #5) while rollout_off was already active caused generation to 
   slow because actor-resident FSDP weights/activations contend with rollout predict on the 
   same GPU. This is a combination effect: in isolation, actor_off was a sweep winner, but 
   when rollout is already resident, the extra GPU memory pressure hurts.

4. **Predict is not the bottleneck for openpi -- only 14.7% of generation time.**
   At baseline, predict_busy_s is 45.58 s out of 310.51 s generation (14.7%). The env (70.6%) 
   dominates. This means further model-level optimizations to predict would have limited impact 
   until the env bottleneck is addressed.

## Memory Profile

The openpi model is memory-efficient. Even with rollout offload disabled and actor offload enabled, 
peak memory is only 49.3 GiB (61.6% of 80 GiB). When both offloads are disabled, peak memory is 
50.45 GiB (63.1%). There is ample headroom for this model size.

## Model-Specific Config Recommendations

| Knob | Recommendation | Rationale |
|------|---------------|-----------|
| `rollout.enable_offload` | **false** | Model is small enough (~7 GB) to keep resident |
| `actor.micro_batch_size` | **32** | 64 causes memory contention during generation |
| `actor.enable_offload` | **true** | Prevents GPU memory contention with rollout predict |

## Cross-References

- [[env/robotwin_adjust_bottle/ppo-openpi]]
- [[algorithm/ppo/robotwin_adjust_bottle-openpi]]
- [[cfg/rollout_offload_disabled/robotwin_adjust_bottle-openpi-ppo]]
- [[knob-effect/enable_offload]]

## Raw Data Summary

```json
{
  "knob_effects": {
    "rollout.enable_offload": {
      "direction": "positive",
      "magnitude": 0.1205,
      "range": ["true", "false"],
      "confidence": "high"
    },
    "actor.micro_batch_size": {
      "direction": "negative",
      "magnitude": -0.0939,
      "range": ["32", "64"],
      "confidence": "medium"
    },
    "actor.enable_offload": {
      "direction": "negative",
      "magnitude": -0.0884,
      "range": ["true", "false"],
      "confidence": "medium"
    }
  },
  "memory": {
    "winner_max_used_gib": 49.29,
    "total_gib": 80,
    "winner_max_used_pct": 61.6,
    "oom_risk": "safe"
  },
  "dominant_bottleneck": "generation (env_interact 70.6%, predict 14.7%)",
  "model_size_gb": 7.0
}
```