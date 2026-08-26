# Atomic Profiling Concepts — per-component RLinf performance modeling

Domain facts the atomic-profile report template draws on. Read this before
interpreting a sweep's JSON output or writing an `atomic-profile-report.md`.

---

## 1. What is atomic profiling?

Atomic profiling is a **per-component, isolation-guaranteed** measurement methodology.
Instead of profiling the entire RL pipeline end-to-end (where component interactions
and phase scheduling mask individual scaling behavior), atomic profiling measures each
worker **alone on its own GPU(s)** with all other workers idle.

| Component | Sweep op | Knob | What it measures |
|-----------|----------|------|-----------------|
| **Env** | `env.interact_step` | `num_envs` | CPU-bound simulator stepping + reward model inference |
| **Rollout** | `rollout.predict` | batch size | GPU-bound model inference (diffusion denoising / LLM generation) |
| **Actor** | `actor.train_micro_batch` | `micro_batch_size` | GPU-bound FSDP training (forward + backward + optimizer) |

Because each component runs on **disjoint GPUs** with no cross-component interference,
the measured latency is the **intrinsic cost** of that component at that knob setting.

---

## 2. Why isolate components?

In a full RL training step, the three components run in a **scheduled pipeline**:

```
Generation phase:  [env→rollout→env→rollout→...]  (actor idle)
Training phase:                                     [actor FWD/BWD/OPT]  (env/rollout offloaded)
```

The generation phase overlaps env and rollout (pipeline parallelism), so the
measured wall time is `max(T_env, T_rollout)` per stage, not `T_env + T_rollout`.
This masks individual component scaling. Atomic profiling fills this gap by
providing **pure per-component cost curves** that explain *why* the pipeline
behaves the way it does:

- **Isolate the bottleneck**: If rollout is the slower component, increasing env
  parallelism won't speed up the step — rollout is on the critical path.
- **Measure CPU saturation cleanly**: The env component's CPU utilization is
  masked by the rollout's GPU activity in a full run.
- **Find the OOM boundary**: Each component's peak memory in isolation tells you
  the exact knob value where it OOMs, independent of other components.

---

## 3. Atomic modeling objectives

The atomic sweep serves four concrete goals that the agent uses to make
optimization decisions:

### a. Get hold of onload/offload overhead
Each component's model weights can be offloaded to CPU when idle and onloaded
when active. The atomic sweep measures the one-shot cost of these transfers.
Knowing `onload_ms` and `offload_ms` lets the agent decide whether offloading
a component is worthwhile — if `(onload + offload) / step_time > 10%`, keeping
the component resident may be better (if VRAM allows).

### b. Master optimal parallelism degree
Each component has a different scaling curve with its knob:
- **Env**: `num_envs` → linear CPU-bound scaling; CPU saturation is the ceiling.
- **Rollout**: batch size → GPU-compute-bound; near-perfect linear scaling;
  VRAM is the ceiling (OOM boundary).
- **Actor**: `micro_batch_size` → U-shaped training cost; too small = sync
  overhead, too large = OOM; sweet spot in the middle.

The atomic sweep finds the shape of each curve so the agent can propose the
optimal knob value for each component independently.

### c. Balance rollout-env single-step time
In the generation phase, env and rollout run in a 2-stage pipeline:
`T_generation = max(T_env_total, T_rollout_total)`. The slower component
determines the generation wall time. The atomic sweep lets the agent compute
this balance and tune the slower side until `T_env ≈ T_rollout`.

### d. Satisfy memory constraints
Each component's peak VRAM from the atomic sweep is the intrinsic footprint.
The agent composes these with the offload plan to ensure per-GPU peak memory
stays within the VRAM budget, using the formula:
`per-GPU peak ≈ max(resident_set, active_offloaded_component)`.

---

## 4. Instance parallelism

Each atomic knob represents a **per-instance** parallelism degree — the amount
of work one worker instance (one GPU card) handles in one operation:

| Component | Per-instance knob | Definition |
|-----------|-------------------|------------|
| **Actor** | `micro_batch_size` | Microbatch size processed on **each actor GPU card** (each actor worker) in one forward/backward micro-step |
| **Env** | `num_envs` | Number of envs stepped on **each env worker** in one `interact_step` |
| **Rollout** | batch size | Number of trajectories inferred on **each rollout worker** in one `predict` call |

The total workload per component is then derived from the instance parallelism
and the number of worker instances:

- **Actor**: `global_batch_size = micro_batch_size × actor_world_size × num_micro_batches`
- **Env**: `total_num_envs = num_envs × env_world_size × pipeline_stage_num`
- **Rollout**: `total_batch = batch_size × rollout_world_size`

Equivalently, the per-worker processing for rollout and env is:

```
per_worker_env = total_num_envs / pipeline_stage_num / env_world_size
per_worker_rollout_batch = total_batch / pipeline_stage_num / rollout_world_size
```

Where `pipeline_stage_num` is the rollout-env pipeline depth (typically 1 or 2).
When `pipeline_stage_num = 2`, env and rollout form a 2-stage pipeline and share
GPU resources across stages.

---

## 5. Sweep methodology

A sweep runs a single component's benchmark function (e.g. `rollout.predict`)
at multiple knob values while keeping all other knobs fixed. Results are stored
in `<log_dir>/bench_msgs/sweep_{env_interact,rollout_predict,actor_micro_batch}.json`.

### 5.1 Timing

- **`timed_vram`**: CUDA-synchronized timer with warmup + repeats. Returns
  `ms_mean`, `ms_std`, `ms_min`, `ms_max`. Warmup iterations (default 2) are
  discarded to exclude JIT / cache fill / cudnn autotune.
- **`timed_once`**: One-shot timer for non-repeatable operations like weight
  onload/offload (model transfer between CPU and GPU).

### 5.2 Resource sampling

A background `_ResourceSampler` thread samples at 5ms intervals during the
timed region. Key metrics: `peak_alloc_MB` (PyTorch tensor VRAM),
`peak_reserved_MB` (caching allocator), `host_rss_peak_MB` (host RAM, env only),
`cpu_percent_mean` (CPU util, env only), `gpu_util_mean` (GPU compute util).

### 5.3 Sweep control

The three sweeps run on **disjoint GPUs** (configurable via
`RLINF_BENCH_ENV_GPUS`, `RLINF_BENCH_ROLLOUT_GPUS`, `RLINF_BENCH_ACTOR_GPUS`).
Sweep points are specified as multipliers of the default knob value
(`RLINF_BENCH_MULTIPLIERS`, default `0.25,0.5,1,2,4`). Before sweeping, the
harness runs one real env↔rollout round-trip ("inline capture") to produce seed
messages for the actor sweep.

---

## 6. Reading the cost curves

### 6.1 Env: `num_envs` → step latency

The env is typically **CPU-bound** (video decoding, OCR, physics simulation).
Key patterns:

- **Linear scaling**: `step_ms ∝ num_envs` — each env consumes the same CPU time.
- **CPU saturation**: When `cpu_percent_mean` plateaus at N×100% (N cores fully
  loaded), the CPU is the bottleneck.
- **GPU util near 0%**: The env uses GPU only for the reward model.
- **Host RSS linear**: Each env adds a fixed amount of host memory.

The production `num_envs` is typically chosen as the largest value that still
fits within the per-step latency budget.

### 6.2 Rollout: batch size → inference latency

The rollout is **GPU compute-bound** (model inference). Key patterns:

- **Linear scaling**: `ms ∝ batch_size` — GPU compute-bound, latency scales
  proportionally.
- **GPU saturation**: `gpu_util_mean ≈ 98–100%`.
- **VRAM scales sub-linearly**: Model weights dominate, activations add modestly.
- **OOM boundary**: The last non-OOM point defines the safe upper bound.

The production batch size is typically the largest safe value under the OOM
ceiling.

### 6.3 Actor: `micro_batch_size` → training latency

The actor is **GPU compute-bound + communication-bound** (FSDP training). Key
patterns:

- **U-shaped cost curve**: Too-small micro-batches are slow due to communication
  overhead; too-large micro-batches OOM.
- **More micro-steps = more overhead**: Each micro-step adds gradient
  synchronization overhead.
- **Large micro-batch = more VRAM**: Activations scale linearly; the OOM boundary
  is where activations exceed available VRAM after the FSDP shard is loaded.

The sweet spot is typically at `micro_batch_size = 4` or `8` for diffusion
models, and `micro_batch_size = 40–80` for LLM/VLA models.

---

## 7. Composing the components

The atomic measurements compose into the total step time:

```
T_step = max(T_env_total, T_rollout_total) + T_actor_train
```

Where:
- `T_env_total = (num_trajectories / num_envs) × T_env_per_step`
- `T_rollout_total = (num_trajectories / batch_size) × T_rollout_per_batch`
- `T_actor_train = T_actor_micro_batch × num_micro_batches`

In the default **sync mode** with `pipeline_stage_num=2`, env and rollout form
a 2-stage pipeline during generation, so `T_generation = max(T_env, T_rollout)`.
In async mode, actor also pipelines with generation.

### Bottleneck identification

| Pattern | Signal | Implication |
|---------|--------|-------------|
| `T_rollout >> T_env` | Rollout-bound | Add rollout GPUs, reduce denoising steps |
| `T_env >> T_rollout` | Env-bound | Increase `total_num_envs`, optimize env code |

---

## 8. Memory & offload model

Each component's peak memory from the atomic sweep is the **intrinsic** footprint.
In production with offload, per-GPU peak memory is:

```
Per-GPU peak ≈ max(resident_set, active_offloaded_component)
```

Where `resident_set` = sum of components with `enable_offload=false` on that GPU,
and `active_offloaded_component` = memory of the component currently using the GPU.

### Onload/Offload cost

Weight transfer between CPU and GPU is measured separately via `timed_once`:

```
offload_cost_pct = (offload_time + onload_time) / step_time
```

If `offload_cost_pct > 10%`, disabling offload for that component may be
worthwhile — but only if VRAM is sufficient to keep it resident.

### Memory scaling (linear model)

All three components exhibit approximately linear memory scaling with their knob:
- **Actor**: `peak ≈ BASE_actor + slope_actor × micro_batch_size`
- **Rollout**: `peak ≈ BASE_rollout + slope_rollout × batch_size`
- **Env**: `peak ≈ BASE_env + slope_env × num_envs`

These coefficients are **model/hardware-specific** — re-fit from the sweep data
rather than guessing. The production memory model (see `concepts.md` §5) composes
these with the offload flags.

---

## 9. When to run atomic profiling

Run the atomic sweep **once per config family**, right after the baseline
evaluation (before any round-1 proposals). It answers:

- **"How much headroom is left?"** — VRAM headroom and latency budget per component.
- **"Which component is the bottleneck?"** — Compare per-trajectory costs.
- **"What's the memory model?"** — Fit slope and intercept for offload decisions.
- **"What's the OOM boundary?"** — Exact knob value where each component OOMs.

The sweep is a **precondition** for whole-pipeline trials, not a replacement.
Proposals should always cite the sweep curve before launching an expensive
multi-GPU trial.

Results are written to `bench_msgs/sweep_<op>.json` (structured) and
`bench_msgs/sweep_<op>.csv` (spreadsheet-friendly), with `is_default` and `oom`
flags per record.