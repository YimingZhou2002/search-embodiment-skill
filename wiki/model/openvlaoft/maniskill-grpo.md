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

# openvlaoft + maniskill + grpo

## Key Findings

### 1. openvlaoft is generation-bound under GRPO

The openvlaoft model, when used with GRPO (no critic network), has a profile where the rollout inference pass (predict + env_interact) is the dominant cost. At baseline, generation accounts for 124.9% of step time (with overlap, the real number is 99.8% at the winning config). The model's forward pass during training (actor_forward ~1.4s per micro-batch) is dwarfed by the rollout inference needed for trajectory generation.

### 2. FSDP activation memory scales with micro_batch_size

Raising `actor.micro_batch_size` from 40 to 80 added 28.9 GiB of FSDP activation memory. This is the model's specific footprint: openvlaoft's activations (under FSDP with gradient checkpointing) are substantial enough that doubling mbs pushes near OOM. Since GRPO is generation-bound, this memory is wasted -- the model's training throughput does not improve.

### 3. openvlaoft responds well to tne amortization

The model's rollout inference cost is partially fixed (model loading, weight sync from actor, offload churn). Doubling tne from 256 to 512 amortized these fixed costs, giving 1.77x step time for 2x the data. The model's predict step scales roughly linearly with batch size (mean predict call: 1.55s at tne=256, 3.35s at tne=512), so the scaling is sub-linear overall.

### 4. Memory pressure is moderate

At baseline, openvlaoft uses 48.2 GiB (60.3%) across 8 GPUs. With tne=512 and env offload off, it uses 61.16 GiB (76.5%). The model itself is not the memory bottleneck -- the env state and trajectory buffers are the main memory consumers at higher tne.

## Cross-References

- [[env/maniskill/grpo-openvlaoft]] -- full campaign report
- [[algorithm/grpo/maniskill-openvlaoft]] -- algorithm-specific insights
- [[cfg/tne_scaling/maniskill-openvlaoft-grpo]] -- tne scaling pattern

## Raw Data Summary

```json
{
  "campaign": "search_20260818-065327",
  "model": "openvlaoft",
  "env": "maniskill",
  "algorithm": "grpo",
  "memory_profile": {
    "baseline_max_used_gib": 48.2,
    "winner_max_used_gib": 61.16,
    "total_gib": 80
  },
  "generation_behavior": {
    "baseline_generation_pct": 124.9,
    "winner_generation_pct": 99.8,
    "actor_idle_pct": 94.9
  },
  "mbs_sensitivity": {
    "mbs_40_vs_80": "no benefit",
    "memory_delta_gib": 28.9
  }
}
```