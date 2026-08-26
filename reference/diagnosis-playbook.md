# Diagnosis Playbook — Pattern → Cause → Fix

Translate `diagnosis.json` signals into diagnoses and fix directions for RLinf
embodiment training. Read this **after** running `diagnose.py`; read
[`concepts.md`](concepts.md) first for the domain model (per-trajectory metric,
generation nesting, offload memory model).

## How to use

For each pattern: **Signals** (exact `diagnosis.json` fields + thresholds),
**Why**, **First-line fix**, **Deeper fixes**, **Exceptions**. Most runs match
2–4 patterns. **Rank by magnitude** — how many seconds of per-trajectory time (or
GiB of OOM headroom) each one is worth — and fix the biggest first. The goal is
always to lower `efficiency.step_time_per_traj_s` (or to stay off the OOM ceiling).

---

## Pattern A — Generation-bound pipeline (actor GPUs idle)

**Signals:**
-  **Async RL**:Only happens on async mode, `mode`=async [rollout,env] and [actor] forms a complete pipeline.
- `pipeline_split.generation_pct` ≫ `training_pct` (reference: 68% vs 27%).
- `pipeline_split.actor_idle_during_generation_pct` high (reference: 68%) — the
  actor's training GPUs are idle for most of the step.

**Why:** generation (rollout + env) and training run back-to-back; while rollout
generates, the actor blocks in `recv_traj`. Half the fleet idles half the step.

**First-line fix:**
- **Raise generation throughput** so the phase shrinks: more `total_num_envs`,
  more rollout replicas(more rollout-env GPU resources) (both are safe throughput knobs — [`concepts.md`](concepts.md) §7).
- **Bigger rollout batch / higher inference concurrency** to fill the rollout GPUs.

**Exceptions:** strictly on-policy algorithms that require fresh weights every
step limit overlap — then shrink generation rather than overlap it.

---

## Pattern B — Rank straggler / env tail effect

**Signals:**
- `rank_straggler.<tag>.straggler_ratio` > ~1.3 for `env_interact_step`,
  `predict`, `actor_forward`, or `actor_backward`.
- Large per-rank spread in `rank_straggler.<tag>.per_rank_mean_s`.

**Why:** the phase waits on its slowest rank. For env, variable episode lengths /
scene complexity make some sims run long; for actor, data-imbalanced micro-batches.

**First-line fix:** balance the work — pack/sort env episodes by expected length
so concurrently-running envs finish together; equalize per-rank batch sizes.

**Deeper fixes:** split long episodes across ranks; classify-and-dispatch (short
episodes on a fast path, long ones chunked).

**Exceptions:** ManiSkill-style fixed-horizon tasks are naturally balanced
(reference run: all ratios ≈ 1.0) — don't chase a tail that isn't there.

---

## Pattern C — Offload tax vs OOM trade-off

**Signals:**
- `pipeline_split.offload_cost_pct` non-trivial (reference: ~6%).
- Cross-reference `memory.max_used_pct` and the per-component offload flags in
  `meta.config`.
- `offload_cost.component.mean_s`> 5% step time.

**Why:** the hybrid engine shuffles params/grads/optimizer between CPU and GPU
each step to fit colocated components. That movement costs wall-clock, but buys
the memory headroom that keeps the run off the OOM ceiling.

**First-line fix (guided by [`concepts.md`](concepts.md) §5):**
- **Keep `rollout.enable_offload=true`** under colocation — turning it off is the
  #1 OOM cause (reference `offload_rollout_false` → 99%, crashed).
- **`env.enable_offload=false`** is often a net win: costs ~7 GiB but removed the
  env onload/offload churn for −10% per-traj in the reference sweep. Do it only if
  `memory.max_used_pct` has enough headroom(more than `env.memory_per_rank`) to make env component resident.
- **`actor.enable_offload`**: low value either way (~2 GiB, slightly slower off) —
  leave at default.

**Exceptions:** disaggregated placement (each component its own GPUs) removes the
need for offload entirely — then offload_cost should be ~0.

---

## Pattern E — Low GPU utilization / colocation bubbles

**Signals:**
- `gpu_utilization.overall.overall_active_gpus.avg_gpu_util` low (reference: 57%).
- `gpu_utilization.low_util_gpus` non-empty — a group of GPUs (usually the
  actor-only ones) idle during generation.

**Why:** the phase serialization of Pattern A shows up as time-averaged idle:
actor-only GPUs idle during generation; rollout GPUs idle during training.

**First-line fix:** same levers as Pattern A (overlap phases / disaggregate). Util
is the symptom; pipeline serialization is the cause — fix the phase structure, not
the number.

**Exceptions:** env-heavy CPU-bound workloads (Pattern F) can bottleneck on CPU
while GPUs legitimately wait — then raising GPU util means fixing the CPU side.

---

## Pattern F — CPU-bound env simulators

**Signals:**
- `cpu_saturation._multicore_flagged` lists env ranks (max_cpu ≫ 200%; reference
  env workers hit ~5600% ≈ 56 cores).
- env `avg_process_gpu_util` ~ 0 while `env_interact_step` is a large share of the
  generation phase (`generation_internals.env_interact_pct_of_generation` high).
- `atomic-profile-report.md` and atomic profile shows that env worker GPU-utilization are below 50%

**Why:** physics/render sim is CPU-side; if it dominates the generation phase, the
step is gated by CPU throughput, not GPU.

**First-line fix:** collocation of env and rollout workers (`cluster.env: all` and `cluster.rollout: all`) will spawn more env worker processes, and spawn more rollout workers than disaggrated placement(worker instance num = component placement GPU num); Setting `pipeline_stage_num: 2` allows env and rollout workers to run on the same device at the same time(when env hardly utilzd GPUs, rollout could use them instead). raise `total_num_envs` so per-step CPU overhead amortizes (Pattern I).


**Exceptions:** if `predict_pct_of_generation` dominates instead, the rollout
*inference* is the gen bottleneck (Pattern H), not env.

---

## Pattern G — GPU-memory headroom & OOM risk

**Signals:**
- `memory.oom_risk` = `elevated` (headroom < 5GB) or `high` (headroom < 1GB).
- In a sweep, `compare_runs.py` OOM table / data-gaps flag it.

**Why / reading (see [`concepts.md`](concepts.md) §5–6, §8):**
- **high + crashed:** a resident component pushed peak to the ceiling — almost
  always `offload_rollout_false` under colocation, or `micro_batch_size` past the
  ~108 knee. Re-enable the offload / lower mbs.
- **elevated but safe:** e.g. `offload_env_false` at 86% — acceptable if it buys
  throughput and headroom remains.
- **lots of headroom** (headroom>10GB): you can raise mbs for efficiency until
  headroom<5GB, staying under the knee.

**First-line fix:** 
1.match the memory model — offload the component consume most GPU memory with least onload/offload time cost(typically rollout and actor, env workers usually take least GPU memory and most onloading cost).
2.reduce data parallism by turning down actor.micro_batch_size and total_num_envs, depending on the OOM rank and component placement.

**Exceptions:** 

---

## Pattern H — Rollout inference throughput

**Signals:**
- `generation_internals.predict_pct_of_generation` high while
  `env_interact_pct_of_generation` is lower.
- rollout `avg_process_gpu_util` modest despite predict dominating.

**Why:** the policy forward (`predict`) is the generation bottleneck — often
batch under-fill (few concurrent envs per rollout step) or an untuned inference
engine.

**First-line fix:** raise inference batch (more envs per rollout worker); tune the
serving engine (paged KV, larger max-batch, dtype); more rollout replicas.

**Exceptions:** if env dominates instead, this is Pattern F.

---

## Pattern I — Too few envs / small batch (poor amortization)

**Signals:**
- Low `meta.config` `total_num_envs` or `micro_batch_size`, and high
  `efficiency.step_time_per_traj_s` relative to peers.
- In a sweep: the low-tne / low-mbs runs sit at the bottom of the ranking
  (reference: tne_32 5.28 s/traj +98%, mbs_20 2.98 +11%).

**Why:** fixed per-step overhead (weight sync, offload, env reset, launch) is
amortized over few trajectories, so per-trajectory cost blows up.

**First-line fix:** raise `total_num_envs` past the knee (reference knee at
64→128) and keep `micro_batch_size` ≥ ~80; both amortize overhead and are memory-
safe with offload on (Pattern G / [`concepts.md`](concepts.md) §7).

**Exceptions:** Do not raise `total_num_envs*rollout_epoch` to more than 4 times of the baseline knob value. The product of these 2 value denotes number of samples generated and trained within one step. Raising too much will affect RL algorithm's integrity and results in degradation of `success_once` standard. memory- or CPU-limited setups cap how far you can push — raise
until `memory.oom_risk` leaves `safe` or env CPU saturates (Pattern F).

---
## Pattern I — Imbalanced Intra Generation Pipeline

**Signals:**
- `stage_breakdown.rollout.predict.mean`/`stage_breakdown.env.env_interact_step.mean`>1.4.
- `stage_breakdown.rollout.predict.mean`/`stage_breakdown.env.env_interact_step.mean`<0.7.

**Why:** In generation phase rollout.predict and env.interact forms a pipeline, imbalanced rollout predict and env interact time will cause bubble within the pipeline and leaving faster component idle waiting for the slower component.

**First-line fix:** 
- assign more resources to the slower and more resource demanding component. e.g. raise the amount of gpus assign to the rollout.
- Try collocate env and rollout, first set `env: 0-7` and `rollout:0-7`, then try set pipeline_stage_num=1 and pipeline_stage_num=2 sequentially.


## Ranking template for the report

Rank directions by **(expected per-trajectory-time reduction) × (1 / effort)**.
Cite the specific `diagnosis.json` numbers as evidence. Stop at 3–5 — more dilutes
the signal.

```
Priority 1: <pattern> — <concrete config/code change>
  Evidence: <metric field = value, e.g. generation_pct=68, actor_idle=68>
  Expected: <Δ per-traj-s or throughput, which knob>
  Effort:   <low / medium / high>
  Why now:  <highest-leverage reason>

Priority 2: ...
```
