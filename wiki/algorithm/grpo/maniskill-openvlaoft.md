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

# grpo + maniskill + openvlaoft

## Key Findings

### 1. GRPO is intrinsically generation-bound

Unlike PPO, which has a critic model adding training-side compute, GRPO uses only an actor policy. This means the training pass (forward + backward + optimizer step) is relatively lightweight -- at baseline, training accounts for only 48.7% of step time, while generation accounts for 124.9% (with overlap). The training pass is not the bottleneck.

The implication: knobs that improve training throughput (`actor.micro_batch_size`, `actor.fsdp_config.gradient_checkpointing` -- already on) have negligible effect on overall step time.

### 2. The only effective lever is amortizing generation overhead

With GRPO's generation-bound profile, the key optimization is to amortize the fixed overhead of generation (model loading, weight sync, offload churn) over more trajectories. Increasing `total_num_envs` from 256 to 512 achieved 1.77x step time for 2x trajectories, yielding -11.3% per-traj improvement.

### 3. GRPO has no critic, so no value loss or advantage computation overhead

The `actor/compute_adv` stage takes only 0.1% of step time (0.073s per step at winning config). The `actor_policy_loss` stage takes 0.41s per micro-batch. These are negligible compared to the rollout generation cost.

### 4. Pipeline parallelism is counterproductive

GRPO with the openvlaoft model uses all 8 GPUs for both actor and rollout. Increasing `rollout.pipeline_stage_num` from 1 to 2 added a pipeline bubble that increased env_interact by 47% and predict by 46%. The sync-mode pipeline does not overlap well with GRPO's generation pattern.

## Cross-References

- [[env/maniskill/grpo-openvlaoft]] -- full campaign report
- [[model/openvlaoft/maniskill-grpo]] -- model-specific insights
- [[cfg/tne_scaling/maniskill-openvlaoft-grpo]] -- tne scaling pattern

## Raw Data Summary

```json
{
  "campaign": "search_20260818-065327",
  "algorithm": "grpo",
  "env": "maniskill",
  "model": "openvlaoft",
  "bottleneck_analysis": {
    "generation_pct": 99.8,
    "training_pct": 64.3,
    "adv_pct": 0.1,
    "weight_sync_pct": 10.0,
    "offload_cost_pct": 33.1
  },
  "effective_knobs": ["total_num_envs", "enable_offload"],
  "ineffective_knobs": ["micro_batch_size", "pipeline_stage_num"]
}
```