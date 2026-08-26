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

# PPO Algorithm: robotwin_adjust_bottle + openpi

## Key Findings

1. **PPO is generation-bound, not training-bound.** 
   At baseline, generation consumes 93.1% of step time while training (forward + backward + optimizer) 
   consumes only 5.7%. Actor GPUs are idle 91.5% of generation time waiting for rollout trajectories. 
   This means config knobs that affect training (mbs, gradient checkpointing, etc.) have limited leverage.

2. **Training sensitivity is minimal.** 
   Actor forward takes ~1.0 s per step, backward ~0.08 s, optimizer step ~0.05 s. The total training 
   time of 19.15 s is dwarfed by generation at 310.51 s. Even a 50% training speedup would only 
   improve overall step time by ~2.9%.

3. **Weight sync is a minor cost.** 
   `sync_model_to_rollout` + `sync_model_from_actor` total ~5.6 s per generation round, accounting 
   for 3.7% of step time. This is not a bottleneck worth optimizing for this combination.

4. **PPO's advantage computation is negligible.** 
   `compute_adv` takes only 0.055 s (0.0% of step time). The lightweight PPO advantage calculation 
   is not a factor.

## Algorithm-Specific Recommendations

| Knob | Recommendation | Rationale |
|------|---------------|-----------|
| All training knobs | **Leave at default** | Training is only 5.7% of step time; changes have minimal impact |
| `actor.global_batch_size` | **1024 (default)** | No evidence that changing this helps |
| `actor.fsdp_config.gradient_checkpointing` | **false (default)** | Training is already fast; checkpointing would add overhead |

## Memory Profile

PPO stores optimizer states and gradients for the policy model. With the openpi model (~3B params), 
the total memory footprint is modest: baseline 38.3 GiB, winner 49.3 GiB, both well within 80 GiB.

## Cross-References

- [[env/robotwin_adjust_bottle/ppo-openpi]]
- [[model/openpi/robotwin_adjust_bottle-ppo]]
- [[cfg/rollout_offload_disabled/robotwin_adjust_bottle-openpi-ppo]]
- [[knob-effect/enable_offload]]

## Raw Data Summary

```json
{
  "pipeline_split_baseline": {
    "generation_pct": 93.1,
    "training_pct": 5.7,
    "weight_sync_pct": 3.7,
    "adv_pct": 0.0,
    "offload_cost_pct": 15.4,
    "actor_idle_during_generation_pct": 91.5
  },
  "training_breakdown": {
    "actor_forward_mean_s": 1.01,
    "actor_backward_mean_s": 0.084,
    "actor_optimizer_step_mean_s": 0.055
  },
  "dominant_bottleneck": "generation",
  "training_sensitivity": "low"
}
```