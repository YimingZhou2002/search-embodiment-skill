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

# Wan2.2-TI2V-5B (Diffusion Video OCR Model)

## Key Findings

### 1. The 5B model dominates memory, leaving almost no headroom

Across all configs, peak GPU memory hovers at ~77.6 GiB / 80 GiB (97%). The model weights are the dominant consumer. Changing env count (192 to 384) or micro_batch_size (4 to 2) barely moves the memory needle. This is a critical constraint: the model is 5B parameters, and with FSDP + gradient checkpointing on 8 GPUs, it still consumes nearly the entire 80 GiB per GPU.

### 2. No training step — pure generation pipeline

This is a diffusion model for text-to-video OCR, not an RL policy. The pipeline has zero training time (`training_s = 0.0`). The entire step time is spent in generation (predict + env_interact). Knobs that affect training (micro_batch_size, gradient checkpointing) have minimal or no effect on performance.

### 3. Offload is mandatory but expensive

The model is too large to keep on GPU at all times. Rollout GPUs offload the model after each generation batch (mean ~11.3s per offload) and reload before predict (mean ~5.4s per onload). Actor GPUs also offload parameters/gradients (~1.25s offload, ~1.17s onload) even though no training occurs. The `offload_auxiliary_modules` knob further worsened performance by 28.3%, as the aux modules are needed during inference.

### 4. Predict is the single largest generation sub-stage

Predict time dominates the generation phase: 73-79% of generation time across configs. The mean predict per rank at node 4 (best) is 27.7s, while env_interact is 26.0s. Both are roughly balanced but predict has a slight edge, suggesting the diffusion inference forward pass is the primary compute demand.

## Cross-References

- [[env/wan22/unknown-ti2v_5b]] — Full campaign report
- [[cfg/tne_scaling/wan22-ti2v_5b-unknown]] — How tne scaling applies to diffusion models
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
    },
    "actor.micro_batch_size": {
      "direction": "negative",
      "magnitude": 0.0109,
      "range": ["2", "4"],
      "best_value": 4,
      "confidence": "medium"
    },
    "actor.model.wan22_ti2v_5b.offload_auxiliary_modules": {
      "direction": "negative",
      "magnitude": 0.0734,
      "range": ["false", "true"],
      "best_value": false,
      "confidence": "high"
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
  "bottleneck_detail": "generation_pct=165.4%, training_pct=0.0%"
}
```