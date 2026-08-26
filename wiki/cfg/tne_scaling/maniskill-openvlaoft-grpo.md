---
dimensions:
  env: maniskill
  model: openvlaoft
  algorithm: grpo
  cfg_pattern: tne_scaling
campaign: search_20260818-065327
created: 2026-08-20
updated: 2026-08-20
confidence: high
baseline_speed: 0.6375
best_speed: 0.5652
speedup: "-11.3%"
num_nodes: 8
---

# tne_scaling + maniskill + openvlaoft + grpo

## Key Findings

### 1. tne scaling shows a clear U-shaped effect

The `total_num_envs` (tne) knob was explored at 256, 512, and 768 (with 1024 SKIPPED). The effect is strongly U-shaped:

| tne | Step Time (s) | Per-Traj (s) | vs Baseline | Memory (GiB) |
|-----|---------------|-------------|-------------|-------------|
| 256 | 163.2 | 0.6375 | -- | 48.2 |
| 512 | 289.4 | 0.5652 | -11.3% | 61.16 |
| 768 | 408.4 | 0.6216 | +10.0% vs 512 | 78.53 |

The sweet spot is tne=512. At tne=256, the fixed generation overhead dominates. At tne=768, training time doubles (185.95 -> 408.36s) and overtakes generation as the bottleneck.

### 2. Sub-linear scaling: 1.77x step time for 2x trajectories

The step time scaling factor from tne=256 to tne=512 is 1.77x (289.4/163.2), which is significantly sub-linear. This means the fixed overhead component (model loading, weight sync, offload churn) is substantial relative to the per-trajectory cost.

### 3. tne scaling requires memory headroom

tne=512 requires ~61 GiB (76.5% of 80 GiB). The baseline with env offload on uses 48.2 GiB. Disabling env offload freed ~5 GiB, providing the critical headroom. Without this, tne=512 would risk OOM.

### 4. tne=768 and tne=1024 are dead ends

tne=768 regressed to 0.6216 s/traj at 78.53 GiB (98.2%). tne=1024 was SKIPPED because it would clearly be worse. The training phase scales roughly 1.5x with trajectory count, so at tne=768, training time (185.95s -> 408.36s) more than doubles and overtakes generation.

## Conditions for effectiveness

This tne scaling pattern is specific to GRPO with openvlaoft on maniskill. The key condition is that the **generation-bound** bottleneck (characteristic of GRPO) allows tne increases to amortize fixed overhead. Under PPO, where training is more balanced, the pattern may differ.

## Cross-References

- [[env/maniskill/grpo-openvlaoft]] -- full campaign report
- [[knob-effect/total_num_envs_scaling]] -- cross-campaign total_num_envs data

## Raw Data Summary

```json
{
  "campaign": "search_20260818-065327",
  "cfg_pattern": "tne_scaling",
  "env": "maniskill",
  "model": "openvlaoft",
  "algorithm": "grpo",
  "tne_values_tested": [256, 512, 768],
  "tne_1024": "SKIPPED",
  "best_tne": 512,
  "scaling_factor_256_to_512": 1.77,
  "memory_at_best_gib": 61.16,
  "memory_headroom_gib": 18.84,
  "prerequisite": "env.train.enable_offload=false"
}
```