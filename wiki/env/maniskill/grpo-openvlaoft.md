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

# maniskill + grpo + openvlaoft

## Key Findings

### 1. Generation is the dominant bottleneck

At baseline (tne=256, env offload on), generation accounts for 124.9% of step time (pipeline overhead means generation and training overlap, but generation dominates). Actor GPUs are idle 115.6% of generation time (offloaded weights must be reloaded). The 48.2% offload tax (78.6s per step) is the single largest overhead category.

At the winning config (tne=512, env offload off), generation is still 99.8% of step time, and actor GPUs remain idle 94.9% of the generation phase. The bottleneck is intrinsic to GRPO with this model: the rollout inference pass (predict + env_interact) dominates because GRPO has no critic model to add training-side compute.

### 2. Doubling total_num_envs from 256 to 512 achieves sub-linear scaling

The winning knob is `env.train.total_num_envs` = 512. Step time increased from 163.2s to 289.4s (1.77x) for 2x the trajectories, yielding the per-traj improvement of 0.6375 -> 0.5652 s/traj (-11.3%). The mechanism is amortization of fixed generation overhead (model loading, weight sync, offload churn) over more trajectories.

### 3. Disabling env offload frees critical memory headroom

`env.train.enable_offload` = false alone saves ~5 GiB of GPU memory (baseline max 48.2 GiB -> 61.16 GiB at winning config, but with much more data). Alone it only improved 1.7% (0.6375 -> 0.6265), but it was essential for enabling the tne=512 increase -- without the freed memory, the 512 trajectories would not fit.

### 4. micro_batch_size=80 is a dead end for GRPO

Raising `actor.micro_batch_size` from 40 to 80 added 28.9 GiB of FSDP activation memory (near OOM at 96.3%) with no performance benefit. GRPO is generation-bound, not training-bound, so training throughput knobs are ineffective.

## Knob Sensitivity Map

| Knob | Range Explored | Direction | Magnitude | Best Value |
|------|---------------|-----------|-----------|------------|
| `env.train.total_num_envs` | 256, 512, 768 | U-shaped (peak at 512) | -11.3% at 512, +10.0% at 768 vs 512 | 512 |
| `env.train.enable_offload` | true, false | Negative (disabling helps) | -1.7% alone, essential enabler for tne=512 | false |
| `actor.micro_batch_size` | 40, 80 | Flat/negative | +1.2% at 80 alone | 40 (keep default) |
| `rollout.pipeline_stage_num` | 1, 2 | Strongly negative | +22.5% at 2 | 1 (keep default) |

## Memory & Bottleneck Profile

| Config | Max GPU Mem (GiB) | % of 80 GiB | OOM Risk | Bottleneck |
|--------|-------------------|-------------|----------|------------|
| Baseline (tne=256, offload on) | 48.2 | 60.3% | Safe | Generation (124.9%) |
| Winner (tne=512, offload off) | 61.16 | 76.5% | Safe | Generation (99.8%) |
| Near OOM (mbs=80, tne=512) | 79.13 | 98.9% | High | Generation |
| Regressed (tne=768, offload off) | 78.53 | 98.2% | Elevated | Training (now dominates) |

The winning config leaves 18.84 GiB of headroom. The memory ceiling on 8x A800-SXM4-80GB is effectively 80 GiB; tne=1024 would certainly OOM (tne=768 already at 98.2%).

## Known Dead Ends

- **tne=768** (Node #7): Increasing from 512 to 768 regressed to 0.6216 s/traj (+10.0%). Training time doubled (185.95 -> 408.36s) and now dominates the step. Memory at 78.53 GiB (98.2%, elevated).
- **tne=1024** (Node #8): SKIPPED -- predicted to OOM based on tne=768 trend.
- **mbs=80** (Node #2): 0.6450 s/traj (+1.2%), near OOM at 77.0 GiB (96.3%). No benefit.
- **mbs=80 + tne=512** (Node #5): 0.5758 s/traj (-9.7% vs baseline), but 79.13 GiB (98.9%) -- not worth the risk.
- **pipeline_stage=2** (Node #4): 0.7807 s/traj (+22.5%). Pipeline bubble increased env_interact by 47% and predict by 46%.

## Cross-References

- [[model/openvlaoft/maniskill-grpo]] -- openvlaoft model-specific insights
- [[algorithm/grpo/maniskill-openvlaoft]] -- GRPO algorithm-specific insights
- [[cfg/tne_scaling/maniskill-openvlaoft-grpo]] -- tne scaling pattern
- [[knob-effect/total_num_envs_scaling]] -- total_num_envs cross-campaign data
- [[knob-effect/enable_offload]] -- enable_offload cross-campaign data
- [[knob-effect/micro_batch_size_tuning]] -- micro_batch_size cross-campaign data
- [[knob-effect/pipeline_stage_tuning]] -- pipeline_stage cross-campaign data

## Raw Data Summary

```json
{
  "campaign": "search_20260818-065327",
  "env": "maniskill",
  "model": "openvlaoft",
  "algorithm": "grpo",
  "gpus": "8x A800-SXM4-80GB",
  "knob_effects": {
    "env.train.total_num_envs": {
      "direction": "u-shaped",
      "magnitude": -0.0723,
      "range": ["256", "512", "768"],
      "best": 512,
      "confidence": "high"
    },
    "env.train.enable_offload": {
      "direction": "negative",
      "magnitude": -0.011,
      "range": ["true", "false"],
      "best": false,
      "confidence": "high"
    },
    "actor.micro_batch_size": {
      "direction": "none",
      "magnitude": 0.0075,
      "range": ["40", "80"],
      "best": 40,
      "confidence": "high"
    },
    "rollout.pipeline_stage_num": {
      "direction": "negative",
      "magnitude": 0.1542,
      "range": ["1", "2"],
      "best": 1,
      "confidence": "high"
    }
  },
  "memory": {
    "baseline": {
      "max_used_gib": 48.2,
      "total_gib": 80,
      "max_used_pct": 60.3,
      "oom_risk": "safe"
    },
    "winner": {
      "max_used_gib": 61.16,
      "total_gib": 80,
      "max_used_pct": 76.5,
      "oom_risk": "safe"
    }
  },
  "dominant_bottleneck": "generation",
  "bottleneck_detail": "generation_pct=99.8, actor_idle_during_generation_pct=94.9",
  "winner_overrides": {
    "env.train.total_num_envs": 512,
    "env.train.enable_offload": false
  },
  "best_objective": 0.5652,
  "baseline_objective": 0.6375,
  "speedup_pct": -11.3
}
```