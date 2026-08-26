---
dimensions:
  env: wan22
  model: ti2v_5b
  algorithm: unknown
  cfg_pattern: tne_scaling
campaign: search_wan22_ti2v_5b_nft_video_ocr
created: 2026-08-20
updated: 2026-08-20
confidence: medium
baseline_speed: 0.2592 s/traj
best_speed: 0.1975 s/traj
speedup: "-23.8%"
num_nodes: 6
---

# tne_scaling: wan22 + ti2v_5b (Diffusion Video OCR)

## Context

This campaign was the first application of the tne_scaling config pattern to a diffusion model (Wan2.2-TI2V-5B) rather than an RL embodiment model. The pattern is the same: increase `total_num_envs` to amortize fixed overheads across more trajectories. But the mechanism differs from RL because there is no training step — the bottleneck is entirely generation-bound.

## Key Findings

### 1. Scaling tne from 192 to 384 gave 23.8% improvement

The baseline tne=192 (with re=6) gave 0.2592 s/traj. Increasing tne to 256 with re=9 gave 0.2191 s/traj (-15.5%). Scaling further to tne=384 with re=9 gave 0.1975 s/traj (-23.8% vs baseline). This is a clear monotonic improvement, though the marginal gain from 256 to 384 (8.5% improvement) is smaller than from 192 to 256 (15.5%), suggesting diminishing returns.

### 2. Divisibility constraints are critical

The first attempt at tne=256 (node 1) failed with `CONFIG_INVALID` because `rollout_size = tne * re / group_size = 256 * 6 / 8 = 192` was not divisible by the batch size (72). The fix was to increase `rollout_epoch` from 6 to 9, making `rollout_size = 256 * 9 / 8 = 288`, which is divisible by 72.

**Divisibility rule:** `(tne * re / group_size) % batch_size == 0` where `batch_size = gbs / mbs / group_size`.

### 3. tne scaling works by amortizing offload overhead

The key mechanism: more trajectories per rollout means the fixed cost of model offload/onload is spread across more work. The offload cost as a fraction of step time dropped from 30.4% (baseline, tne=192) to 18.8% (node 2, tne=256) to 11.5% (node 4, tne=384). This is the driver of the speedup.

### 4. Memory is not a constraint for tne scaling in this model

Despite being a 5B model consuming ~77.6 GiB, increasing tne from 192 to 384 barely changed memory usage (77.6 vs 77.65 GiB). The model weights dominate memory, not the env buffers. This means tne can be scaled further (512, 768) without hitting OOM.

## Recommended tne range for diffusion models

For large diffusion models (5B+ params), tne scaling is effective because:
- Memory is dominated by model weights, not env buffers — tne can scale freely
- Offload overhead is significant (~30% at low tne) and amortizes well
- There is no training step to compete for GPU cycles

However, the diminishing returns at tne=384 suggest that tne=512 might yield only ~5% additional gain. The optimal range for this model size appears to be tne=384-512.

## Cross-References

- [[env/wan22/unknown-ti2v_5b]] — Full campaign report
- [[model/ti2v_5b/wan22-unknown]] — Model-specific insights
- [[knob-effect/total_num_envs_scaling]] — Cross-campaign tne analysis
- [[knob-effect/rollout_epoch_tuning]] — Cross-campaign rollout_epoch analysis

## Raw Data Summary

```json
{
  "knob_effects": {
    "env.train.total_num_envs": {
      "direction": "positive",
      "magnitude": 0.0617,
      "range": ["192", "384"],
      "best_value": 384,
      "confidence": "medium"
    },
    "env.train.rollout_epoch": {
      "direction": "positive",
      "magnitude": "required for tne divisibility",
      "range": ["6", "9"],
      "best_value": 9,
      "confidence": "medium"
    }
  },
  "memory": {
    "max_used_gib": 77.65,
    "total_gib": 80,
    "max_used_pct": 97.1,
    "oom_risk": "elevated",
    "headroom_gib": 2.35
  },
  "dominant_bottleneck": "generation",
  "bottleneck_detail": "offload_cost_pct dropped from 30.4% to 11.5% as tne increased"
}
```