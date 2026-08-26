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

# robotwin_adjust_bottle + PPO + openpi

## Key Findings

1. **The SAPIEN env is the hard floor -- 70.6% of generation time and serial.** 
   At 256 envs, `env_interact_busy_s` consumes 219.2 s out of 310.5 s of generation time (70.6%). 
   The env processes use ~1.4 CPU cores on average (`cpu_mean~136%`), and doubling `total_num_envs` to 512 
   doubled `env_interact_busy_s` from 219 to 452 s -- perfectly linear, confirming zero parallelism to unlock 
   via config. The remaining ~0.82 s/traj is the env floor. No config knob can fix this.

2. **`rollout.enable_offload=false` gives a clean 9.2% speedup.** 
   The openpi rollout model is only ~7 GB (atomic sweep: `onload peak_alloc~7.0 GB`), so offloading it 
   every step was pure churn. Offload cost collapsed from 51.3 s to 7.6 s (15.4% -> 2.5% of step time). 
   Peak memory rose from 38.3 GiB to 49.3 GiB (47.8% -> 61.6% of 80 GiB), still safe. 
   This is the opposite of the openvla pattern where the rollout model is ~25 GB and offload is essential.

3. **Actor-side knobs backfire when stacked on rollout_off.** 
   Disabling actor offload (node #5, 1.2710 s/traj, -2.5% vs winner) causes generation to slow due to 
   actor-resident weights/activations contending with rollout `predict` on the same GPU. 
   Raising micro_batch_size to 64 (node #6, 1.2765 s/traj, -2.0% vs winner) creates a memory side-effect 
   that slows generation despite a small training speedup.

4. **`pipeline_stage_num=2` hurts predict efficiency.** 
   Predict calls double (80->160) and each call slows (1.05->1.49 s mean). The overlap benefit 
   cannot offset the smaller-batch predict inefficiency. Result: +1.2% vs baseline.

## Knob Sensitivity Map

| Knob | Range Explored | Direction | Magnitude | Best Value | Confidence |
|------|---------------|-----------|-----------|------------|------------|
| `rollout.enable_offload` | true -> false | **positive** | **-0.1205 s/traj (-9.2%)** | **false** | high |
| `env.train.total_num_envs` | 256 -> 512 | negative | +0.3182 s/traj (+24.4%) | 256 (default) | high |
| `actor.enable_offload` | true -> false | negative | +0.0884 s/traj (+7.5% vs winner) | true (default) | medium |
| `actor.micro_batch_size` | 32 -> 64 | negative | +0.0939 s/traj (+7.9% vs winner) | 32 (default) | medium |
| `rollout.pipeline_stage_num` | 1 -> 2 | negative | +0.0159 s/traj (+1.2%) | 1 (default) | medium |
| `env.train.enable_offload` | true -> false | neutral | +0.0120 s/traj (+0.9%, noise) | either | low |

## Memory & Bottleneck Profile

**Baseline (node #0):** Generation is the dominant bottleneck at 93.1% of step time. Training is only 5.7%, 
weight sync 3.7%. Actor GPUs are idle 91.5% of generation time. Peak GPU memory is 38.27 GiB (47.8% of 80 GiB), 
headroom 41.73 GiB, OOM risk safe.

**Winner (node #3, rollout_off=false):** Generation slightly higher at 95.5% (offload cost shrinks from 15.4% to 2.5%, 
so generation's share of the pie increases). Peak memory rises to 49.29 GiB (61.6%), headroom 30.71 GiB, still safe.

**tne=512 (node #4):** Peak memory hits 57.68 GiB (72.1%), headroom 22.32 GiB, still safe but 
approaching the OOM wall. Generation time doubles due to the env bottleneck.

## Known Dead Ends

- **`total_num_envs=512` (+24.4%):** The env scales linearly with tne because the SAPIEN simulation is 
  serial per env worker. Doubling envs from 256 to 512 doubled `env_interact_busy_s` from 219 to 452 s. 
  Never increase tne for this env.
- **`rollout.pipeline_stage_num=2` (+1.2%):** Predict calls double (80->160) and each predict slows 
  (1.05->1.49 s mean). The overlap benefit is negligible.
- **`actor.enable_offload=false` stacked on rollout_off (+7.5% vs winner):** Actor-resident weights 
  contend with rollout predict on the same GPU. The isolated sweep showed this as a winner, but it 
  backfires in combination.
- **`actor.micro_batch_size=64` stacked on rollout_off (+7.9% vs winner):** Memory side-effect during 
  generation outweighs the small training speedup.

## Cross-References

- [[model/openpi/robotwin_adjust_bottle-ppo]]
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
    "env.train.total_num_envs": {
      "direction": "negative",
      "magnitude": -0.3182,
      "range": ["256", "512"],
      "confidence": "high"
    },
    "actor.enable_offload": {
      "direction": "negative",
      "magnitude": -0.0884,
      "range": ["true", "false"],
      "confidence": "medium"
    },
    "actor.micro_batch_size": {
      "direction": "negative",
      "magnitude": -0.0939,
      "range": ["32", "64"],
      "confidence": "medium"
    },
    "rollout.pipeline_stage_num": {
      "direction": "negative",
      "magnitude": -0.0159,
      "range": ["1", "2"],
      "confidence": "medium"
    },
    "env.train.enable_offload": {
      "direction": "none",
      "magnitude": -0.0120,
      "range": ["true", "false"],
      "confidence": "low"
    }
  },
  "memory": {
    "baseline_max_used_gib": 38.27,
    "winner_max_used_gib": 49.29,
    "total_gib": 80,
    "baseline_max_used_pct": 47.8,
    "winner_max_used_pct": 61.6,
    "oom_risk": "safe"
  },
  "dominant_bottleneck": "generation",
  "bottleneck_detail": "env_interact 70.6% of generation, predict 14.7% of generation"
}
```