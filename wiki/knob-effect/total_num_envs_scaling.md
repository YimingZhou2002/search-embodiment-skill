---
dimensions:
  knob: total_num_envs
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

# total_num_envs Scaling

## Campaign: maniskill + PPO + openvlaoft (search_20260813-070815)

This is the first campaign contributing data to this knob-effect entry. Initial findings below.

## Summary

`env.train.total_num_envs` controls the number of parallel environment instances used for trajectory generation. Increasing tne amortizes fixed costs (offload, weight sync, model initialization) across more trajectories, reducing the per-traj time. The effect is monotonic but diminishing, and the best value depends on the collocation state and rollout_epoch.

## Data

**Disaggregated branch (env_offload=false, tne explored from 128 to 320):**
- 128 -> 192: 1.0106 -> 0.8160 (-19.3%)
- 192 -> 224: 0.8160 -> 0.7985 (-2.1%)
- 224 -> 256: 0.7985 -> 0.7556 (-5.4%)
- 256 -> 288: 0.7556 -> 0.7426 (-1.7%)
- 288 -> 320: 0.7426 -> 0.7278 (-2.0%)
- 320 -> 384: 0.7278 -> 0.6933 (-4.7%) [with env_offload=true]

**Collocated branch (env_offload=false, tne explored from 320 to 512):**
- 320 -> 384: 0.5695 -> 0.5570 (-2.2%)
- 384 -> 448: 0.5570 -> 0.5516 (-1.0%)
- 448 -> 512: 0.5516 -> 0.5359 (-2.8%)

**With re=2 (collocated):**
- 384 (re=2): 0.5173 (best)
- 448 (re=2): 0.5181 (0.2% worse — within noise)

## Key Insight

tne and rollout_epoch are **substitutes** for amortizing fixed costs. At re=2, tne=384 (0.5173) and tne=448 (0.5181) are statistically identical. The best tne value is 384 — high enough to amortize but not so high that it adds unnecessary generation+training work. Once re=2 amortizes the fixed cost, more envs add work in exact proportion to the extra trajectories.

## Best Value

**384**, under collocation + re=3. The best overall config (tne=384, re=3, collocated) achieves 0.5050 s/traj.

## Conditions

- tne scaling is more effective under disaggregation (-19.3% for 128->192) than under collocation (-2.2% for 320->384), because disaggregation has higher fixed costs to amortize.
- Under collocation with re>=2, tne past 384 gives no meaningful benefit.
- No OOM risk from tne alone — the memory ceiling is hit by mbs=80, not by tne.

## Raw Data Summary

```json
{
  "knob": "env.train.total_num_envs",
  "range_explored": [128, 192, 224, 256, 288, 320, 384, 448, 512],
  "best_value": 384,
  "direction": "positive (diminishing)",
  "confidence": "high",
  "data_points": {
    "disaggregated": [
      {"tne": 128, "speed": 1.0106},
      {"tne": 192, "speed": 0.8160},
      {"tne": 224, "speed": 0.7985},
      {"tne": 256, "speed": 0.7556},
      {"tne": 288, "speed": 0.7426},
      {"tne": 320, "speed": 0.7278}
    ],
    "collocated": [
      {"tne": 320, "speed": 0.5695},
      {"tne": 384, "speed": 0.5570},
      {"tne": 448, "speed": 0.5516},
      {"tne": 512, "speed": 0.5359}
    ],
    "collocated_re_2": [
      {"tne": 384, "speed": 0.5173},
      {"tne": 448, "speed": 0.5181}
    ]
  },
  "notes": "tne and re are substitutes. Best paired with collocation + re=3."
}
```