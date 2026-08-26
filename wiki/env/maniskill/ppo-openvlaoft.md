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

# maniskill + PPO + openvlaoft

## Key Findings

### 1. Generation is the dominant bottleneck, and collocation eliminates cross-card transfer

The best node's diagnosis shows generation accounts for 62.4% of step time (784.1 s out of 1256.9 s), with the actor idle 62.2% of the time waiting for rollout trajectories. Training is only 29.7%. The **single largest lever** was collocating env and rollout onto the actor's 8 GPUs (component_placement.env=0-7, component_placement.rollout=0-7), which cut per-traj from 0.7278 s to 0.5695 s at tne=320 — a **-21.7%** improvement. This works by eliminating cross-card data transfer and reusing the actor's resident weights for rollout, so rollout no longer needs its own weight copy.

### 2. rollout_epoch amortizes fixed offload/sync costs, capped at 3 by a runtime bug

Increasing rollout_epoch from 1 to 2 saved 7.1% (0.5570 to 0.5173 s), and from 2 to 3 saved a further 2.4% (0.5173 to 0.5050 s). The mechanism: offload cost (5.4% of step time) and weight-sync cost (1.2%) are fixed per rollout step, so spreading them across more epochs reduces per-traj overhead. `rollout_epoch=4` crashes with a `pybind11_object_dealloc` error — a runtime object-lifecycle bug, not OOM — so the ceiling is hard at 3.

### 3. tne scaling and rollout_epoch are substitutes, not complements

At re=1, tne 384->448->512 gives diminishing returns (-1 to -3% per step). At re=2, tne=448 (0.5181 s) is statistically identical to tne=384 (0.5173 s). Once rollout_epoch amortizes the fixed costs, more envs add generation+train work in exact proportion to the extra trajectories. The best tne value is 384 — high enough to amortize but not so high that it adds unnecessary work.

### 4. env_offload is a no-op under collocation; mbs=80 is a memory-wall dead end

Under collocation, re-enabling env_offload (#21 vs #15) leaves peak memory byte-identical (77.94 GiB / 97.4%) and only adds onload overhead. The env sim is tiny relative to model weights. mbs=80 reaches 97.4-99.0% memory (#15/#21/#22) and is dominated by the safe re=2/re=3 branch at ~64% memory. pipeline_stage_num=2 is strictly harmful (+11.7% alone, +2.3% in disaggregated).

## Knob Sensitivity Map

| Knob | Range Explored | Direction | Magnitude | Best Value |
|------|---------------|-----------|-----------|------------|
| `cluster.component_placement.env` + `.rollout` | 0-3 / 4-7 -> 0-7 / 0-7 | positive | -21.7% at tne=320 | **0-7 (collocate with actor)** |
| `env.train.rollout_epoch` | 1, 2, 3, 4 | positive (1->3) | -7.1% (1->2), -2.4% (2->3) | **3** |
| `env.train.total_num_envs` | 128, 192, 224, 256, 288, 320, 384, 448, 512 | positive (diminishing) | -19.3% (128->192), flattens past 384 | **384** |
| `env.train.enable_offload` | true, false | no-op under collocation | ~0% under collocation; -2% disaggregated | **false** |
| `actor.micro_batch_size` | 40, 80 | negative (80 hits memory wall) | -0.2% to -0.3% at safe ranges | **40** |
| `rollout.pipeline_stage_num` | 1, 2 | strictly harmful | +11.7% alone | **1** |

## Memory & Bottleneck Profile

- **Memory (best node #23):** 50.79 GiB / 80 GiB = 63.5% — safe, with 29.21 GiB headroom
- **OOM risk:** safe (best config). High for mbs=80 combos (97.4-99.0%).
- **Dominant bottleneck:** Generation (62.4% of step time). Actor idle 62.2% during generation.
- **Bottleneck detail:** `generation_s: 784.126 / step_time_s: 1256.855`. Training 29.7%, offload 5.4%, weight sync 1.2%.
- **GPU utilization:** 78.7% average, 100% peak — good utilization.

## Known Dead Ends

1. **rollout_epoch=4 (#24):** CRASH with `pybind11_object_dealloc(): Tried to deallocate unregistered instance!` — a runtime object-lifecycle bug, not OOM. Do not attempt re=4 without code changes.
2. **mbs=80 (#8, #15, #21, #22):** OOM at tne=256 (#8) and 97.4-99.0% memory utilization under collocation (#15 97.4%, #21 97.4%, #22 99.0%). The memory wall is terminal — doubling micro_batch_size from 40 to 80 adds ~15 GiB peak memory.
3. **pipeline_stage_num=2 (#2, #10):** Strictly harmful. +11.7% alone (#2), +2.3% in the disaggregated tne=192 branch (#10). Pipeline parallelism adds sync overhead that outweighs any benefit for this model size.
4. **tne past 384:** Diminishing returns. At re=1, tne 384->448->512 gives -1 to -3% each. At re=2, tne=448 is identical to tne=384. Further tne increases add generation+training work without amortization benefit.
5. **env_offload=true under collocation:** No memory benefit, adds onload overhead (+2.1%). The env sim is CPU-bound and tiny relative to model weights.

## Cross-References

- [[knob-effect/total_num_envs_scaling]] — tne scaling details
- [[knob-effect/rollout_epoch_tuning]] — rollout_epoch amortization
- [[knob-effect/enable_offload]] — offload behavior
- [[knob-effect/micro_batch_size_tuning]] — mbs memory wall
- [[knob-effect/pipeline_stage_tuning]] — pipeline stage failure
- [[cfg/collocation/maniskill-openvlaoft-ppo]] — collocation pattern
- [[model/openvlaoft/maniskill-ppo]] — openvlaoft model specifics
- [[algorithm/ppo/maniskill-openvlaoft]] — PPO algorithm specifics

## Raw Data Summary

```json
{
  "knob_effects": {
    "cluster.component_placement": {
      "direction": "positive",
      "magnitude": 0.1583,
      "range": ["disaggregated (0-3/4-7)", "collocated (0-7/0-7)"],
      "confidence": "high"
    },
    "env.train.rollout_epoch": {
      "direction": "positive",
      "magnitude": 0.052,
      "range": ["1", "3"],
      "confidence": "high"
    },
    "env.train.total_num_envs": {
      "direction": "positive",
      "magnitude": 0.2536,
      "range": ["128", "384"],
      "confidence": "high"
    },
    "env.train.enable_offload": {
      "direction": "none",
      "magnitude": 0.0,
      "range": ["true", "false"],
      "confidence": "high"
    },
    "actor.micro_batch_size": {
      "direction": "negative",
      "magnitude": 0.005,
      "range": ["40", "80"],
      "confidence": "high"
    },
    "rollout.pipeline_stage_num": {
      "direction": "negative",
      "magnitude": -0.1187,
      "range": ["1", "2"],
      "confidence": "high"
    }
  },
  "memory": {
    "max_used_gib": 50.79,
    "total_gib": 80,
    "max_used_pct": 63.5,
    "oom_risk": "safe"
  },
  "dominant_bottleneck": "generation",
  "bottleneck_detail": "generation_s: 784.126 / step_time_s: 1256.855 (62.4%); actor idle 62.2% during generation"
}
```