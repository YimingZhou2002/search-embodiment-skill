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

# Collocation: maniskill + openvlaoft + PPO

## Config Pattern

Collocation places `env` and `rollout` components on the same GPUs as `actor` (all on 0-7), rather than on separate dedicated GPU pairs (env=0-3, rollout=4-7). This is a cluster-level placement change, not a model-level change.

**Baseline (disaggregated):** `actor=0-7, env=0-3, rollout=4-7`
**Collocated:** `actor=0-7, env=0-7, rollout=0-7`

## Why Was It Tried?

The disaggregated baseline had env and rollout each on 4 dedicated GPUs, requiring cross-card data transfer between components. The actor was idle 62% of the time waiting for rollout trajectories. Collocation was hypothesized to reduce cross-card transfer overhead and allow the rollout to reuse the actor's resident weights, which would be especially beneficial for the large openvlaoft model.

## Observed Effect

**Collocation is the single largest lever, giving -21.7% at tne=320.**

At the disaggregated tne=320 baseline (#7: 0.7278 s/traj), collocating env+rollout with actor (#12) dropped to 0.5695 s/traj — a 0.1583 s (21.7%) reduction. This is bigger than any other single-knob effect in the campaign.

The mechanism: under disaggregation, rollout runs on 4 GPUs with its own weight copy. Collocating onto 8 GPUs means the rollout can use the actor's weights directly, eliminating cross-card weight transfer and doubling the GPU count for rollout. The env sim is tiny and collocating it has no memory impact.

## Conditions for Effectiveness

- **Works best when the model is large enough that weight copies are expensive.** openvlaoft is a VLA model; having a separate rollout weight copy on 4 GPUs under disaggregation is costly to synchronize.
- **Works best when the actor has memory headroom.** Under the best config, memory is 63.5% (50.79 GiB / 80 GiB), leaving 29.21 GiB for the collocated env/rollout. The env sim is CPU-bound and adds negligible GPU memory.
- **env_offload becomes a no-op under collocation.** The env sim is tiny relative to model weights, so offloading it has no memory benefit and only adds onload overhead.
- **Does not require algorithm changes.** Collocation is purely a cluster placement change — it works with standard PPO.

## Raw Data Summary

```json
{
  "pattern": "collocation",
  "baseline_config": {
    "actor": "0-7",
    "env": "0-3",
    "rollout": "4-7"
  },
  "collocated_config": {
    "actor": "0-7",
    "env": "0-7",
    "rollout": "0-7"
  },
  "effect": {
    "at_tne_320": {
      "disaggregated": 0.7278,
      "collocated": 0.5695,
      "improvement_pct": -21.7
    },
    "at_tne_384_re_2": {
      "disaggregated_n_a": null,
      "collocated": 0.5173
    },
    "best_overall": {
      "config": "collocate + tne=384 + re=3",
      "speed": 0.5050,
      "improvement_pct": -50.0
    }
  },
  "memory_impact": "negligible — env sim is CPU-bound, model weights dominate memory",
  "conditions": "large model + actor memory headroom + no env_offload needed"
}
```