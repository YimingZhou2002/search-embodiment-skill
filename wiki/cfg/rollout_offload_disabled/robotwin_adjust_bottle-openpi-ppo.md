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

# Config Pattern: rollout_offload_disabled

## What Was Tried

Setting `rollout.enable_offload=false` to keep the openpi rollout model resident on GPU 
instead of offloading it to CPU after each generation round.

## Why It Was Tried

The openpi model is only ~3B params (~7 GB on GPU). The existing concepts.md guidance 
warned that "rollout offload is the #1 OOM cause" -- but that was written for openvla (7B, ~25 GB). 
For openpi, the rollout model is small enough to keep resident, and the offload/reload overhead 
was expected to be pure churn.

## Observed Effect

- **Step time:** 1.3031 -> 1.1826 s/traj (**-9.2%**)
- **Offload cost:** 51.3 s -> 7.6 s (15.4% -> 2.5% of step time)
- **Peak memory:** 38.27 GiB -> 49.29 GiB (47.8% -> 61.6% of 80 GiB)
- **OOM risk:** Safe (30.71 GiB headroom remaining)

The offload cost collapsed dramatically. The remaining 7.6 s of offload cost is entirely from 
actor offload cycles (which should remain enabled).

## When This Pattern Works

This pattern works when:
- The **rollout model is small** relative to GPU memory (openpi ~7 GB on 80 GiB GPUs)
- The **actor offload is still enabled** (disabling it causes contention -- see node #5)
- There is **sufficient memory headroom** after keeping the rollout model resident

This pattern does NOT work when:
- The rollout model is large (openvla ~25 GB -- offload is essential)
- GPU memory is tight (e.g., 40 GB GPUs or multi-model sharing)

## Comparison with OpenVLA

This is a critical cross-model distinction. The search-embodiment-skill's concepts.md says 
"rollout offload is the #1 OOM cause" -- but that is openvla-specific. For openpi, 
**rollout offload is the #1 performance drain**. Future agents must check the model size 
before applying offload advice.

## Interaction with Other Knobs

- **Stacking with actor_off (`actor.enable_offload=false`):** Negative. Actor-resident weights 
  contend with rollout predict on the same GPU. Regresses 7.5% from the winner.
- **Stacking with mbs=64:** Negative. Extra memory for larger micro-batches slows generation 
  via GPU memory pressure. Regresses 7.9% from the winner.
- **Stacking with tne=512:** Not tested as a combination, but tne=512 was already a dead end 
  (+24.4%) due to the env bottleneck.

## Cross-References

- [[env/robotwin_adjust_bottle/ppo-openpi]]
- [[model/openpi/robotwin_adjust_bottle-ppo]]
- [[algorithm/ppo/robotwin_adjust_bottle-openpi]]
- [[knob-effect/enable_offload]]

## Raw Data Summary

```json
{
  "knob_effect": {
    "rollout.enable_offload": {
      "direction": "positive",
      "magnitude": 0.1205,
      "range": ["true", "false"],
      "confidence": "high"
    }
  },
  "before_after": {
    "baseline_step_time_per_traj": 1.3031,
    "winner_step_time_per_traj": 1.1826,
    "baseline_offload_cost_s": 51.341,
    "winner_offload_cost_s": 7.639,
    "baseline_offload_cost_pct": 15.4,
    "winner_offload_cost_pct": 2.5,
    "baseline_peak_memory_gib": 38.27,
    "winner_peak_memory_gib": 49.29,
    "baseline_peak_memory_pct": 47.8,
    "winner_peak_memory_pct": 61.6
  },
  "applicable_conditions": [
    "rollout model < 10 GB",
    "GPU memory > 40 GB",
    "actor offload remains enabled"
  ]
}
```