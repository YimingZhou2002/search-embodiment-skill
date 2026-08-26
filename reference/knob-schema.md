# Knob schema — what the search may propose

The tunable knobs for target experiment, their legal domains, and the
**hard divisibility invariants** every proposal must satisfy.
`helpers/preflight.py` enforces all of this before any GPU time is spent; keep
this doc and `preflight.BASELINE_KNOBS` / `TUNABLE_KNOBS` in sync.

Overrides are hydra `key=value` tokens passed via `EXTRA_OVERRIDES` to
`run_embodiment.sh` (booleans lowercase: `true`/`false`).

## Tunable knobs

| Knob (hydra key) | Domain | Lever |
|---|---|---|
| `actor.micro_batch_size` | must satisfy `actor.global_batch_size % (mbs·actor_world)==0` | memory (≈`BASE+slope·mbs` GiB) + throughput |
| `env.train.total_num_envs` | `%env_world==0`, `(tne/env_world)%stage==0`, `tne/env_world/stage≥1` → with env_world=4, stage=2: **multiples of 8** | throughput (more trajectories) |
| `env.train.rollout_epoch` | int ≥1 | throughput (data per iter) |
| `rollout.pipeline_stage_num` | int 1 or 2;  | throughput (rollout-env pipelining) |
| `rollout.enable_offload` | bool | memory |
| `env.train.enable_offload` | bool | memory  |
| `actor.enable_offload` | bool | memory  |
| `cluster.component_placement.actor` | contiguous `"a-b"` in 0–7; sets actor_world | placement / parallelism |
| `cluster.component_placement.env` | contiguous `"a-b"`; sets env_world | placement (changes tne divisibility) |
| `cluster.component_placement.rollout` | contiguous `"a-b"`; sets rollout_world | placement |

## Fixed knobs
`actor.global_batch_size` and `algorithm.group_size` are **fixed** for
this recipe (not proposed) but participate in the constraints below.

## Hard invariants (preflight rejects violations)

Let `actor_world`, `env_world`, `rollout_world` = #GPUs in the actor / env / rollout placement ranges.

1. `actor.global_batch_size % (micro_batch_size · actor_world) == 0`
2. `total_num_envs % env_world == 0`
3. `(total_num_envs / env_world) % pipeline_stage_num == 0`
4. `total_num_envs / env_world / pipeline_stage_num ≥ 1`
5. `(total_num_envs / env_world / pipeline_stage_num) % group_size == 0`
6. placement ranges are contiguous `"a-b"`, `0 ≤ a ≤ b ≤ 7`.
7. `(total_num_envs / env_world / pipeline_stage_num) % rollout_world== 0`
8. `(total_num_envs / env_world / pipeline_stage_num) % actor_world== 0`
9. `total_num_envs * rollout_epoch ≤ 8 × baseline_product` (preflight enforces using the baseline from `--config`)

**Coupling to watch:** changing `env` placement changes `env_world` → must
re-check tne (invariants 2–5). Changing `actor` placement changes `actor_world`
→ must re-check mbs (invariant 1). Changing `pipeline_stage_num` → re-check tne.
Always run `preflight.py` on the *resolved* (cumulative) config, not the delta.

## Known effects (from the batch sweep — see `concepts.md`)

- **`env.train.enable_offload=false`** → onloading and offloading envs often comes with substantial cost (30 ~ 100s), but accounts for only a small proportion of GPU memory, thus disabling env offload is typically a good choice.
- **`rollout.enable_offload=false` and `actor.enable_offload=false`** → actor and rollout models both consumes substantial GPU memory, 
It's almost impossible to not offloading rollout and actor components when actor and rollout shares a proportion of GPU sets. And not offloading these two components comes with limit gain in wall time shrink. 
Only disable these two offloads when (a) memory profile in `diagnose.json` shows enough GPU memory capacity to hold both rollout and actor model 
and (b) offload time profile in `diagnose.json` shows offloading time accounts more than 10% of wall time.
- **`total_num_envs`**: too few is very inefficient, too few will also cause instability and fluctuations in `success_once` training increse process;