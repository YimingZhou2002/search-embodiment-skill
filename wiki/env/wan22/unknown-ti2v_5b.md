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

# wan22 + ti2v_5b (Diffusion Video OCR)

## Key Findings

### 1. Generation is the sole bottleneck — training time is zero

Across all nodes, the pipeline shows zero training time (`training_s = 0.0`, `training_pct = 0.0`). This is a diffusion model used for video OCR, not an RL embodiment task — there is no policy update step. The entire step time is generation (155-171% of step time, because actor GPUs idle during generation and the measured span exceeds the step time denominator). Actor GPUs are idle 148-163% of generation time, indicating severe underutilization.

The generation workload splits roughly evenly between `predict` (73-79% of generation time) and `env_interact` (59-73% of generation time), with predict slightly ahead across all configs.

### 2. Increasing total_num_envs reduces per-traj time via amortization

The dominant lever is `env.train.total_num_envs` (tne). Increasing tne from 192 to 384 (2x) while scaling `rollout_epoch` to maintain divisibility reduced per-traj time by 23.8%:

| Config | tne | re | tne*re/gs | s/traj | vs baseline |
|--------|-----|----|-----------|--------|-------------|
| baseline | 192 | 6 | 144 | 0.2592 | -- |
| node 2 | 256 | 9 | 288 | 0.2191 | -15.5% |
| **node 4** | **384** | **9** | **432** | **0.1975** | **-23.8%** |

The mechanism: more envs means more trajectories per rollout, which amortizes the fixed overhead of model offload/onload (actor offload ~1.25s, rollout offload ~11.3s per batch) and improves throughput. The offload cost as a fraction of step time dropped from 30.4% (baseline) to 11.5% (node 4).

### 3. Memory is saturated by the 5B model weights — nearly flat across env counts

Peak GPU memory is ~77.6 GiB / 80 GiB (97%) across all configurations. Changing tne from 192 to 384 barely moves memory (77.6 vs 77.65 GiB). The 5B parameter model weights dominate the memory footprint. There is only ~2.4 GiB headroom, placing the system at elevated OOM risk.

### 4. offload_auxiliary_modules is harmful — frees 1.2 GiB but increases time by 28.3%

Enabling `actor.model.wan22_ti2v_5b.offload_auxiliary_modules=true` reduced peak memory from 77.6 to 76.45 GiB (freed ~1.2 GiB, 95.6% utilization) but increased per-traj time from 0.2592 to 0.3326s (+28.3%). The predict time nearly doubled (mean predict per rank: 32.4s vs 13.2s baseline), suggesting the aux modules are needed during inference and their offload/onload overhead is severe.

### 5. micro_batch_size=2 is net negative

Applied on top of the best config (node 4, tne=384, re=9), reducing mbs from 4 to 2 increased per-traj time from 0.1975 to 0.2084s (+5.5%). Training time increased 20.6% (98.2 to 118.4s) without freeing GPU memory (77.57 GiB, essentially unchanged). Since there is no training step in this diffusion model, the mbs knob has no meaningful effect beyond overhead.

## Knob Sensitivity Map

| Knob | Range Explored | Direction | Magnitude | Best Value | Confidence |
|------|---------------|-----------|-----------|------------|------------|
| `env.train.total_num_envs` | 192, 256, 384 | positive | -23.8% (192 to 384) | 384 | medium |
| `env.train.rollout_epoch` | 6, 9 | positive | scaled with tne | 9 | medium |
| `actor.micro_batch_size` | 4, 2 | negative | +5.5% | 4 | medium |
| `actor.model.wan22_ti2v_5b.offload_auxiliary_modules` | false, true | negative | +28.3% | false | high |

## Memory & Bottleneck Profile

- **Peak memory:** 77.65 GiB / 80 GiB (97.1%) at node 4 (tne=384, re=9)
- **OOM risk:** elevated across all configurations
- **Bottleneck:** generation-bound (165% of step time), with model predict and env_interact roughly balanced
- **GPU utilization:** ~40-50% average, peaking at 100% — plenty of headroom
- **Offload cost:** 11.5% of step time at best config (down from 30.4% at baseline)

## Known Dead Ends

1. **`tne=256` alone (node 1, tag: tne_256)** — Failed with `CONFIG_INVALID`. The rollout size `tne * re / group_size = 256 * 6 / 8 = 192` is not divisible by batch_size (72). Required pairing with `rollout_epoch=9` to make `288 % 72 = 0`.
   - **Divisibility rule:** `(tne * re / group_size) % batch_size == 0` must hold, where `batch_size = gbs / mbs / group_size` (here 576/4/8=18 per GPU, 18*4=72 total).

2. **`offload_auxiliary_modules=true` (node 3)** — Freed 1.2 GiB memory but added 28.3% to per-traj time. The aux modules appear to be needed during inference, so offloading/onloading them per step is expensive. Not worth the memory savings unless OOM is imminent.

3. **`micro_batch_size=2` (node 5)** — Increased per-traj time by 5.5% without freeing memory. Since this is a diffusion model with no training step, mbs has no real effect. The overhead increase comes from more micro-batches through the same total workload.

## Cross-References

- [[model/ti2v_5b/wan22-unknown]] — Model-specific insights for the Wan2.2-TI2V-5B
- [[cfg/tne_scaling/wan22-ti2v_5b-unknown]] — Config pattern analysis
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
  "bottleneck_detail": "generation_pct=165.4%, actor_idle_pct=162.9%, training_pct=0.0%"
}
```