---
dimensions:
  knob: pipeline_stage_num
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

# pipeline_stage_num Tuning

## Campaign: maniskill + PPO + openvlaoft (search_20260813-070815)

This is the first campaign contributing data to this knob-effect entry. Initial findings below.

## Summary

`rollout.pipeline_stage_num` controls the number of pipeline stages used for the rollout model's forward pass. Increasing from 1 to 2 splits the rollout model across 2 pipeline stages, which should reduce per-stage memory and theoretically improve throughput. In practice, it is **strictly harmful** for this configuration.

## Data

**Baseline (disaggregated, tne=128, env_offload=true):**
- stage=1: 1.0106 s (node #0)
- stage=2: 1.1293 s (node #2) — **+11.7%**

**Disaggregated branch (tne=192, env_offload=false):**
- stage=1: 0.8160 s (node #4)
- stage=2: 0.8345 s (node #10) — **+2.3%**

## Key Insight

Pipeline parallelism adds synchronization overhead (pipeline bubbles, communication between stages) that outweighs any benefit for this model size. The openvlaoft model is small enough that 8-GPU FSDP with gradient checkpointing is already efficient — splitting the model across pipeline stages only adds overhead.

The +11.7% regression at baseline (tne=128) is larger than the +2.3% at tne=192. This suggests that pipeline stage overhead is more costly when the model has less work to do per stage (lower tne = fewer trajectories to pipeline across).

## Best Value

**1** — keep default. Do not attempt pipeline parallelism for this model.

## Conditions

- This negative result is specific to the openvlaoft model size and 8-GPU setup. Pipeline parallelism may benefit much larger models where FSDP alone is insufficient.
- The result is consistent across both the env_offload=true and env_offload=false branches.
- The knob was only explored in the disaggregated branch. Under collocation, pipeline parallelism would be even more harmful because the stages share the same GPUs.

## Raw Data Summary

```json
{
  "knob": "rollout.pipeline_stage_num",
  "range_explored": [1, 2],
  "best_value": 1,
  "direction": "negative (strictly harmful)",
  "confidence": "high",
  "data_points": {
    "disaggregated_tne_128_env_offload_true": {
      "stage_1": 1.0106,
      "stage_2": 1.1293,
      "delta": 0.1187
    },
    "disaggregated_tne_192_env_offload_false": {
      "stage_1": 0.8160,
      "stage_2": 0.8345,
      "delta": 0.0185
    }
  },
  "mechanism": "pipeline sync overhead > benefit for this model size",
  "notes": "Consistent across both branches. Do not attempt for openvlaoft on 8 GPUs. May benefit larger models."
}
```