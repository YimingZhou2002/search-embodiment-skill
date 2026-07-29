# Concepts — reading an RLinf embodiment profile

Domain facts the diagnosis playbook and report template both lean on. Read this
once before interpreting a `diagnosis.json`.

---

## 1. Metric of merit: per-trajectory time, NOT Step Time

Judge a run by **`step_time_per_traj_s = Step Time / num_trajectories`** (lower =
faster), never by absolute Step Time.

High-throughput configs (more envs, more replicas) have a *larger* absolute Step
Time but produce proportionally more trajectories, so they amortize fixed
overhead better and are actually **faster per trajectory**. Ranking by Step Time
alone flags the highest-throughput config as the "slowest" — exactly backwards.

`diagnose.py` emits this as `efficiency.step_time_per_traj_s`. Source:
`metrics.log` `Step Time:` + `num_trajectories=`.

`<success_once>` denotes the accuracy and performance of the model, a quick rise (achieving the same success_once score at a earlier time) is a sign of good performance.
---

## 2. The RL pipeline and its phases

Components (Ray workers), each with its own timeline file `<comp>_rank<N>.jsonl`:

| component | role | GPU |
|-----------|------|-----|
| `runner`  | orchestrator | CPU only |
| `actor`   | FSDP policy training (forward/backward/opt) | all training GPUs |
| `rollout` | action generation / inference (SGLang/vLLM) | rollout GPUs |
| `env`     | simulator stepping | CPU-heavy, tiny GPU |

A step has two dominant wall-clock phases that run **back-to-back** (not
overlapped) in the default hybrid/colocated layout:

1. **Generation** — rollout generates actions while env steps the sim. The actor
   is **blocked** here (its `recv_traj` span ≈ the whole generation phase), so
   the actor's GPUs sit idle (on async mode).
2. **Training** — actor does forward/backward/optimizer over the collected
   trajectories. Rollout/env are idle (and usually offloaded).

`diagnose.py` `pipeline_split` reports each phase as a % of Step Time, plus
`actor_idle_during_generation_pct` and `offload_cost_pct`.

### 2.1 Run Modes

There are two kinds of run modes in embodied RLinf *sync Mode* and *async Mode*. Run Modes are determined by specific RLinf Runners used, `run_embodiment.sh` associate to sync runners, `run_async.sh` associate to async runners. Sync Mode blocks actor when rollout is executed, and trains actor after a full set of trajectories is produced, and then updates rollout model according to actor. Therefore rollout-env and actor typically take all GPUs on seperate time. Async Mode allows rollout-actor pipeline, rollout and actor workers can be executed simultaneously. However, in async mode rollout model produce trajectories based on earlier version of the model, and actor updates the rollout model routinely, therefore the trajectories used to train the actor may not be produced using the latest actor model, thus overall accuracy may degraded and flucuate more heavliy during the rise.

---

## 3. `generate_rollouts` is a PARENT span — never double-count

`generate_rollouts` (timeline tag `rollout/generate`) is the **entire generation
phase**. Inside it, nested by timestamp, are:

- `predict` — one rollout policy forward per action step (hundreds per gen).
- `env_interact_step` — one env sim step per action step.

`predict` and `env_interact_step` are **children** of `generate_rollouts`, and
they overlap each other (rollout predicts while env steps). So:

- Do **not** add `predict` + `env_interact` into the top-level phase split — they
  are already inside `generation`.
- Their busy-time percentages are **of the generation phase** and can sum to
  >100% because they run concurrently (an overlap signal, not an error).

`diagnose.py` computes nesting generically by interval containment (`depth` field
in `stage_breakdown`), so it stays correct if RLinf renames/adds tags.

---

## 4. Training has no single wrapper span

There is usually no `run_training` span in the timeline. Training-compute time is
the **union of the actor micro-op tags**: `actor_forward`, `actor_backward`,
`actor_policy_loss`, `actor_optimizer_step` (32 micro-batches each in the
reference run). `diagnose.py` merges these for `pipeline_split.training_s`. This
runs slightly below the metrics `actor/run_training` value, which also includes
optimizer/param loads.

---

## 5. Offload / phase-stagger memory model

Each component has an `enable_offload` flag. When on, the component is moved to
CPU during the phase it isn't active in, so on any GPU:

> **per-rank GPU peak ≈ sum of (the peaks of the *resident (non-offloaded)*
> components) + max(the peak of the active(offloaded) components)  on that GPU** — because per rank GPU peak= resident component + peak offloaded components when they are swapped in.

Golden rule:
per-rank GPU peak ≈ sum(resident (non-offloaded) components)+max(the peak of the active(offloaded) components)

Consequences (an example of maniskill_ppo_openvla):

- **baseline** (all offload on): actor-train (≈61 GiB) and rollout (≈25 GiB) are
  colocated on GPUs 4–7 but **phase-staggered**, so peak ≈ 62 GiB (77%), not 86.
- **`offload_rollout_false`**: rollout (25 GiB) stays resident during training →
  peak ≈ 79 GiB (**99%**, OOM). Under colocation, rollout offload is
  **non-negotiable**.
- **`offload_env_false`**: env resident costs ~7 GiB/GPU (62→69, 86%) but is the
  best single-knob throughput win (−10%) — a "memory-for-throughput" trade.
- **`offload_actor_false`**: ≈ +2 GiB only, and slightly *slower* — least useful
  knob.

Component GPU memory consumption are model-env dependent, which means they varies depending on specific model used(e.g. model: openvla,GR00T,openpi etc. env: maniskill,libero,etc). Determine these parameters actual value based on `gpu_utilization.per-process` section before put them into the report.

`diagnose.py` reports `memory.max_used_pct` and an `oom_risk` band
(safe <5GB left at peak ≤ elevated <1GB left at peak ≤ high).

---

## 6. micro_batch_size ↔ activation memory (linear)

Fixing everything else, actor peak memory is linear in `micro_batch_size`:

```
actor_peak_GiB ≈ BASE + slope × micro_batch_size      (A800-80GB, reference model)
```

Intercept BASE = FSDP shard + optimizer state (mbs-independent); slope =
activations. 

The coefficients are model/hardware-specific — re-fit from a mbs sweep; the
*shape* (linear, hard knee) is the transferable fact.

---

## 7. tne / replicas are pure throughput knobs (no extra peak memory)

With offload on, raising `total_num_envs` (32→256) or replicas (1→3) does **not**
raise single-GPU peak memory (env is offloaded; replicas add GPUs, not memory).
They are safe throughput amplifiers. But **too few** envs is very inefficient:
per-trajectory time explodes at low tne (reference: tne_32 = 5.28 s/traj,
+98% vs baseline) because fixed per-step overhead isn't amortized. There is a
knee — in the reference sweep, tne 64→128 dropped per-traj from 3.26→2.61 s.

---

## 8. Missing `metrics.log` = probable OOM before step 1

A run directory that has **nvitop and timeline samples but no `metrics.log`** almost always means the job crashed/OOM'd before completing the first step (nvitop and timeline samples from process start; metrics are written only after a step finishes). `diagnose.py`sets `efficiency.likely_oom_before_first_step = true` for this case, and `compare_runs.py` surfaces it in the data-gaps table. A run subdir with **no `timeline/` at all** is even earlier failure (didn't produce traces).
Make sure to checkout the `run_embodiment.log` failure traceback to determine the actutal failure cause.
---

## 9. Warmup exclusion

The first **2 occurrences per rank** of a repeated tag (`predict`,
`env_interact_step`, `actor_forward`, …) are warmup (contains onload, JIT, cache fill, cudnn
autotune) and are dropped before averaging. `diagnose.py` does this in
`rank_straggler`.
