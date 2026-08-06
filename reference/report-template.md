# Report Template — RLinf embodiment run

The report is the deliverable; `diagnosis.json` + the plots are evidence. Save as
`<log_dir>/REPORT.md`. A busy reader should get the headline (per-trajectory time +
the one bottleneck) in 30 seconds, then be able to drill down.

Fill every `<...>` from `diagnosis.json` (and the plots). **Cite specific numbers
for every claim** — "generation is 68% of the step (`generation_pct=68.3`)", not
"generation is high". Match patterns using [`diagnosis-playbook.md`](diagnosis-playbook.md)
and the model in [`concepts.md`](concepts.md).

---

## Template

```markdown
# Embodiment Profiling Report — `<config_name>`

**Run:** `<log_dir>`
**Config name:** `<config-name>`
**Model:** `<model-name, e.g. openpi, openvla>`
**Env:** `<sim-env-name, e.g. maniskill, libero,robotwin>`
**Algorithm:** `<algorithm-name, e.g. ppo, grpo>`
**Run Mode:** `<sync/async>`
**Config knobs:** `<fill-in according to log-dir/tensorboard/config.yaml>`
cluster:
  num_nodes: 1
  component_placement:
    actor: 0-7
    env: 0-3
    rollout: 4-7
env:
  train:
    rollout_epoch: 1
    total_num_envs: 128
    enable_offload: False
rollout:
  enable_offload: True
  pipeline_stage_num: 2
actor:
  micro_batch_size: 80
  global_batch_size: 640
  enable_offload: True
**GPUs:** `<N>× <gpu name>` (`<total_gib>` GiB each)
**Global step:** `<efficiency.global_step>/<total_steps>`   
**Profile date:** YYYY-MM-DD

---
## 0. (Optional) Failure analysis
> If profile run fails present this section with a failure analysis, please read `run_embodiment.log` to determine the probable cause of the crash(e.g. divisiablity fault, OOM etc.),  cite log sections that can tell the failure mode, recommend a posssible restoration approach.
The profile run fails due to OOM on rank * and actor/rollout/env worker, recommend enabling rollout offload.
Or: The profile run fails due to divisiablity faults, recommend set *knob to <value>. 

## 1. Headline

> Metric of merit = per-trajectory time (lower = faster). See concepts.md §1.

| Metric | Value | Source |
|---|---:|---|
| **Per-trajectory time** | **`<step_time_per_traj_s>` s** | `Step Time / num_trajectories` |
| Step Time | `<step_time_s>` s | metrics.log |
| Success Once | `<success_once>` s | metrics.log |
| Trajectories | `<num_trajectories>` | metrics.log |
| Generation phase | `<generation_pct>` % of step | pipeline_split |
| Training phase | `<training_pct>` % of step | pipeline_split |
| Actor idle during generation | `<actor_idle_during_generation_pct>` % | pipeline_split |
| Overall GPU util (avg) | `<avg_gpu_util>` % | nvitop_summary |
| Peak GPU memory | `<max_used_gib>` GiB (`<max_used_pct>`%, `<oom_risk>`) | nvitop_summary |

**One-line read:** <the punchline — e.g. "Generation-bound: 68% of the step is
rollout+env while the actor's GPUs idle (Pattern A); per-trajectory time is gated
by generation throughput, not training.">

---

## 2. Per-dimension analysis
> Walk the dimensions; cite `diagnosis.json` fields; state the finding + matched pattern.

### 2.1 Pipeline balance (phases)
<generation vs training vs weight-sync vs adv %; actor-idle-during-generation;
which phase gates the step. Pattern A / E.>

### 2.2 Generation internals
<within generation: predict_pct_of_generation vs env_interact_pct_of_generation;
note they overlap (can exceed 100%); is the gen bottleneck rollout inference
(Pattern H) or env sim (Pattern F)?>

### 2.3 Rank balance / tail
<straggler ratios for env_interact_step / predict / actor_forward|backward;
slowest rank; tail effect present or naturally balanced. Pattern B.>

### 2.4 Memory modeling & Offload analysis
<offload_cost_pct; peak mem vs total and oom_risk; per-component offload flags vs the memory model (concepts.md §5-6); is any knob near the OOM knee? Pattern C / G.>

| Component | enable_offload | time_per_rank| memory_per_rank |
|---|---|---|---|
| env | `true` | 50s | 7.2G |
| rollout | `false` | - | 17.2G |
| actor | `true` | 50s | 20.2G |

### 2.5 GPU utilization
<overall avg util; low_util_gpus group and why (phase serialization). Pattern E.>

### 2.6 CPU / env simulators
<multicore-flagged env ranks and core counts; is the step CPU-gated? Pattern F.>

---

## 3. Summary diagnosis

| Factor | Signal | Status | Impact |
|---|---|---|---|
| <e.g. Generation-bound> | `generation_pct=68`, `actor_idle=68` | 🔴 dominant | highest |
| <Offload/OOM> | `oom_risk=safe`, `max_used_pct=78` | 🟢 ok | — |
| <Rank tail> | `straggler_ratio≈1.0` | 🟢 balanced | — |
| ... | ... | ... | ... |

---

## 4. Optimization directions (ranked by per-trajectory-time impact)

> Each: concrete change, evidence (cite numbers), expected magnitude, effort.
> Rank by (Δ per-traj-s) × (1/effort). Stop at 3–5.

### Priority 1 — <one-line name> (Pattern <X>)
<what to change concretely — which config knob / code path>
- **Evidence:** `<field=value>`, `<field=value>`
- **Expected:** <Δ per-traj-s / throughput, which metric moves>
- **Effort:** <low/medium/high>

### Priority 2 — ...

### Priority 3 — ...

---

## 5. Confidence & data gaps

- **Sure about:** <claims with strong multi-signal support>
- **Uncertain:** <what's ambiguous + what run/data would resolve it>
- **Data gaps:** <from `data_gaps`; note if metrics.log missing ⇒ possible OOM;
  single-step run ⇒ no cross-step variance>

---

## 6. Reproduction & artifacts

    # re-profile (launch with profiling enabled)
    # source profiler/enable_timeline.sh && source profiler/enable_nvitop.sh && \
    #   RUN_LOG_DIR=<log_dir> bash examples/embodiment/run_embodiment.sh <config_name>
    # re-diagnose
    python helpers/diagnose.py <log_dir>

Artifacts in `<log_dir>/`:
- `REPORT.md` (this file)
- `timeline.png` / `timeline.html` — phase Gantt
- `nvitop.png` / `nvitop.html` — resource curves
- `diagnosis.json` / `diagnosis.txt` — extracted signals
```

---

## Style rules

- **Per-trajectory, not absolute.** Lead with `step_time_per_traj_s`; Step Time is
  context, not the verdict.
- **Cite the field.** Every claim names the `diagnosis.json` value behind it.
- **Rank by magnitude.** Fix the 40%-of-step problem before the 2% one.
- **Respect nesting.** Never present `predict` + `env_interact` as top-level phases
  (concepts.md §3); they're inside generation and overlap.
- **Missing data is a finding.** No `metrics.log` ⇒ flag likely-OOM, don't omit.
- **Link, don't dump.** Reference the plots / `diagnosis.json`; keep prose tight.

## Anti-patterns

- ❌ Ranking configs by absolute Step Time.
- ❌ Generic advice with no cited number ("consider more parallelism").
- ❌ Summing nested spans (double-counting generation's children).
- ❌ >5 priorities — you're padding.
- ❌ Omitting the config knobs — nobody can act on the report without them.
