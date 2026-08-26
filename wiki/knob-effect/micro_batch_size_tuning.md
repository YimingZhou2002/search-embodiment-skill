---
dimensions:
  knob: micro_batch_size
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

# micro_batch_size Tuning

## Campaign: maniskill + PPO + openvlaoft (search_20260813-070815)

This is the first campaign contributing data to this knob-effect entry. Initial findings below.

## Summary

`actor.micro_batch_size` controls the per-GPU batch size for the forward/backward pass during training. Increasing mbs should theoretically improve GPU utilization, but for openvlaoft it hits a hard memory wall at 80. The baseline value of 40 is the safe choice.

## Data

**Disaggregated branch, lower tne:**
- mbs=40: 0.9902 s (node #1, env_offload=false, tne=128)
- mbs=80: 0.9933 s (node #3, env_offload=false, tne=128) — **+0.3% (worse)**

**Disaggregated branch, tne=224:**
- mbs=40: 0.7985 s (node #9)
- mbs=80: 0.8026 s (node #13) — **+0.5% (worse, within noise)**

**Disaggregated branch, tne=256:**
- mbs=40: 0.7556 s (node #5)
- mbs=80: **OOM** (node #8) — memory wall

**Collocated branch, tne=320:**
- mbs=40: 0.5695 s (node #12)
- mbs=80: 0.5442 s (node #15) — **-4.4%** but at **97.4% memory**

**Collocated branch, tne=384:**
- mbs=40: 0.5570 s (node #16)
- mbs=80: 0.5505 s (node #22) — **-1.2%** but at **99.0% memory**

## Key Insight

**mbs=80 is a memory-wall dead-end.** Under collocation at tne=320, mbs=80 gives 0.5442 s (97.4% memory, node #15) vs mbs=40 at 0.5695 s. The 4.4% improvement is meaningful, but the memory utilization is terminal — there's no room for further tuning. At tne=384, mbs=80 hits 99.0% memory (node #22). Under disaggregation, mbs=80 at tne=256 is outright OOM (node #8).

The reason: openvlaoft's VLA transformer has large activation memory. Doubling mbs from 40 to 80 adds ~15 GiB peak memory, pushing from a safe 50.8 GiB (63.5%) to 77.9 GiB (97.4%). The remaining 2.1 GiB headroom is insufficient for any further workload.

The mbs=80 branch is dominated by the safe re=2/re=3 branch at ~50.8 GiB (63.5%) with better speed (0.5050 vs 0.5442 s).

## Best Value

**40** — the default. mbs=80 gives marginal speed improvement at the cost of terminal memory pressure.

## Conditions

- openvlaoft's activation memory is the bottleneck, not compute. The model is large enough that doubling mbs causes a ~15 GiB memory spike.
- Under collocation, mbs=80 is briefly competitive but memory-terminal. Under disaggregation, it OOMs at moderate tne.
- Do not attempt mbs=80 unless the model size is reduced or GPU memory is increased.

## Raw Data Summary

```json
{
  "knob": "actor.micro_batch_size",
  "range_explored": [40, 80],
  "best_value": 40,
  "direction": "negative (80 hits memory wall)",
  "confidence": "high",
  "data_points": {
    "disaggregated_tne_128": {
      "mbs_40": 0.9902,
      "mbs_80": 0.9933
    },
    "disaggregated_tne_224": {
      "mbs_40": 0.7985,
      "mbs_80": 0.8026
    },
    "disaggregated_tne_256": {
      "mbs_40": 0.7556,
      "mbs_80": "OOM"
    },
    "collocated_tne_320": {
      "mbs_40": 0.5695,
      "mbs_80": 0.5442,
      "mbs_80_memory_pct": 97.4
    },
    "collocated_tne_384": {
      "mbs_40": 0.5570,
      "mbs_80": 0.5505,
      "mbs_80_memory_pct": 99.0
    }
  },
  "memory_impact": "mbs=40 uses 50.8 GiB (63.5%); mbs=80 uses 77.9 GiB (97.4%) — ~15 GiB increase",
  "notes": "mbs=80 is a dead end. The 4.4% speed improvement at 97.4% memory is not worth the risk. Dominated by safe re=2/re=3 branch."
}
```