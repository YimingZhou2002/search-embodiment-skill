---
dimensions:
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

# openvlaoft: maniskill + PPO

## Key Findings

### 1. openvlaoft is memory-modest for its size — 50.8 GiB peak under collocation

The openvlaoft model with FSDP + gradient checkpointing uses 50.79 GiB / 80 GiB (63.5%) in the best config, leaving 29.21 GiB headroom. This is well below the OOM wall, which is reached only with mbs=80 (97.4-99.0% utilization). The model's memory footprint is dominated by the transformer weights; offload of parameters and optimizer states is needed to keep the actor within bounds (actor.enable_offload=true, rollout.enable_offload=true), but the env offload is unnecessary.

### 2. micro_batch_size 40 is the safety ceiling; 80 hits the wall

Unlike smaller models where mbs=80 might be feasible, openvlaoft's per-rank memory jumps ~15 GiB when doubling mbs from 40 to 80. This is likely due to the large activation memory from the VLA transformer's sequence length. The mbs=80 branch hit 97.4-99.0% memory across all trials and was dominated by the safe re=2/re=3 branch.

### 3. Collocation is especially effective — rollout reuses the actor's weights

Because openvlaoft is a large VLA model, having separate weight copies for rollout under disaggregation has a high memory and transfer cost. Collocation eliminates this entirely: rollout and actor share the same model weights on the same GPUs, removing the need for cross-card weight transfer. This is a bigger win for openvlaoft than it would be for a smaller model.

### 4. Generation dominates over training

The model spends 62.4% of its time in generation (predict/forward), vs 29.7% in training. The predict step averages 2.3 s per call, and env_interact_step averages 7.3 s. This suggests openvlaoft's forward pass through the VLA is compute-bound rather than memory-bound.

## Cross-References

- [[env/maniskill/ppo-openvlaoft]] — main campaign entry
- [[algorithm/ppo/maniskill-openvlaoft]] — PPO algorithm specifics
- [[cfg/collocation/maniskill-openvlaoft-ppo]] — collocation pattern
- [[knob-effect/micro_batch_size_tuning]] — mbs memory wall

## Raw Data Summary

```json
{
  "memory_profile": {
    "best_config_gib": 50.79,
    "total_gib": 80,
    "best_config_pct": 63.5,
    "oom_risk": "safe",
    "mbs_80_memory_pct": 97.4,
    "headroom_gib": 29.21
  },
  "bottleneck": {
    "generation_pct": 62.4,
    "training_pct": 29.7,
    "offload_pct": 5.4,
    "weight_sync_pct": 1.2
  },
  "knob_effects": {
    "actor.micro_batch_size": {
      "direction": "negative beyond 40",
      "magnitude": 0.0,
      "range": ["40", "80"],
      "confidence": "high"
    }
  },
  "model_specific_notes": "FSDP + gradient checkpointing enabled. Collocation especially effective — eliminates weight copy overhead. mbs=80 hits OOM."
}
```