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

# PPO: maniskill + openvlaoft

## Key Findings

### 1. PPO is generation-bound, not training-bound

The PPO training loop is dominated by rollout generation (62.4% of step time, 784.1 s out of 1256.9 s), while training (forward + backward + optimizer) accounts for only 29.7%. The actor is idle 62.2% of the time waiting for rollout trajectories. This means the algorithm's primary bottleneck is inference throughput, not training throughput — any optimization that speeds generation (collocation, tne scaling) is more impactful than training-side optimizations.

### 2. rollout_epoch amortization is uniquely effective for PPO's weight-sync pattern

PPO requires synchronizing the rollout model weights with the actor after each training step. This weight sync (1.2% of step time, 15.6 s) and the associated offload cost (5.4%, 67.8 s) are fixed per rollout step. Increasing rollout_epoch from 1 to 3 amortizes these costs over more epochs, giving a total -9.4% improvement. This is a PPO-specific lever — GRPO or other algorithms with different weight-sync patterns would not benefit.

### 3. PPO's group_size=1 means no intra-batch advantage computation

The baseline uses group_size=1, which means advantage computation is trivial (0.1 s, <0.01% of step time). For models with group_size > 1, the advantage computation would be a larger share of the profile, but here it's negligible.

### 4. No benefit from pipeline parallelism

PPO with openvlaoft does not benefit from pipeline_stage_num=2. The +11.7% regression (#2) suggests the model is small enough that 8-GPU FSDP is already efficient, and adding pipeline stages only adds communication overhead. PPO's training loop also has synchronization points (weight sync to rollout) that limit pipeline utilization.

### 5. tne and re are substitutes for PPO's amortization

Both tne scaling and rollout_epoch increase the number of trajectories per step, amortizing the same fixed costs. The search found they are substitutes — at re=2, tne=384 and tne=448 give the same result (0.5173 vs 0.5181 s). This is because PPO's fixed cost (offload + weight sync) is moderate; once it's amortized by either knob, more trajectories just add proportional work.

## Cross-References

- [[env/maniskill/ppo-openvlaoft]] — main campaign entry
- [[model/openvlaoft/maniskill-ppo]] — openvlaoft model specifics
- [[cfg/collocation/maniskill-openvlaoft-ppo]] — collocation pattern
- [[knob-effect/rollout_epoch_tuning]] — rollout_epoch amortization
- [[knob-effect/total_num_envs_scaling]] — tne scaling

## Raw Data Summary

```json
{
  "algorithm_specific": {
    "group_size": 1,
    "advantage_computation_s": 0.095,
    "advantage_pct": 0.0
  },
  "bottleneck": {
    "generation_pct": 62.4,
    "training_pct": 29.7,
    "actor_idle_pct": 62.2,
    "weight_sync_pct": 1.2,
    "offload_cost_pct": 5.4
  },
  "knob_effects": {
    "env.train.rollout_epoch": {
      "direction": "positive",
      "magnitude": 0.052,
      "range": ["1", "3"],
      "confidence": "high",
      "mechanism": "amortizes fixed weight-sync + offload cost"
    },
    "rollout.pipeline_stage_num": {
      "direction": "negative",
      "magnitude": -0.1187,
      "range": ["1", "2"],
      "confidence": "high",
      "mechanism": "adds sync overhead, model too small for pipeline benefit"
    }
  },
  "algorithm_notes": "PPO is generation-bound, not training-bound. rollout_epoch is a PPO-specific lever. tne and re are substitutes."
}
```