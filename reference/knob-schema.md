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
| `rollout.pipeline_stage_num` | int 1 or 2;  | throughput/memory (rollout pipelining) |
| `rollout.enable_offload` | bool | memory () |
| `env.train.enable_offload` | bool | memory  |
| `actor.enable_offload` | bool | memory  |
| `cluster.component_placement.actor` | contiguous `"a-b"` in 0–7; sets actor_world | placement / parallelism |
| `cluster.component_placement.env` | contiguous `"a-b"`; sets env_world | placement (changes tne divisibility) |
| `cluster.component_placement.rollout` | contiguous `"a-b"`; sets rollout_world | placement |

## Fixed knobs
`actor.global_batch_size` and `algorithm.group_size` are **fixed** for
this recipe (not proposed) but participate in the constraints below.

## Hard invariants (preflight rejects violations)

Let `actor_world`, `env_world` = #GPUs in the actor / env placement ranges.

1. `actor.global_batch_size % (micro_batch_size · actor_world) == 0`
2. `total_num_envs % env_world == 0`
3. `(total_num_envs / env_world) % pipeline_stage_num == 0`
4. `total_num_envs / env_world / pipeline_stage_num ≥ 1`
5. `(total_num_envs / env_world / pipeline_stage_num) % group_size(1) == 0`
6. placement ranges are contiguous `"a-b"`, `0 ≤ a ≤ b ≤ 7`.

**Coupling to watch:** changing `env` placement changes `env_world` → must
re-check tne (invariants 2–5). Changing `actor` placement changes `actor_world`
→ must re-check mbs (invariant 1). Changing `pipeline_stage_num` → re-check tne.
Always run `preflight.py` on the *resolved* (cumulative) config, not the delta.

## Known effects (from the batch sweep — see `concepts.md`)

- **`env.train.enable_offload=false`** → −10% per-traj, +7 GiB (safe if headroom).
- **`rollout.enable_offload=false`** → OOM under default colocation (99% peak). Only
  viable if rollout is disaggregated onto its own GPUs first.
- **`total_num_envs`**: too few is very inefficient (tne_32 +98%); knee around 128;
  256 best. Memory-safe with offload on.
- **`micro_batch_size`**: ≤80 keeps peak ~77%; OOM knee ≈108; below ~40 slower.
- **`actor.enable_offload=false`**: ~+2 GiB, slightly slower — low value.
- **placement allshared** (`env`/`rollout` = "0-7"): slower (+17%) due to contention.
