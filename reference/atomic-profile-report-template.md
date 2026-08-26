# Atomic Profile Report Template — RLinf per-component sweep

The atomic profile report is the deliverable for a per-component scaling sweep
(launched via `benchmark_sweep.py`). It measures each worker in isolation on
**disjoint GPUs**, sweeping its primary scaling knob while holding all other knobs
fixed. The result is a set of cost curves (latency, VRAM, host RSS, CPU/GPU util)
for the **env**, **rollout**, and **actor** components.

Save as `<log_dir>/bench_msgs/atomic-profile-report.md`. A busy reader should get
the bottleneck hierarchy and the recommended production config in 30 seconds, then
drill into per-component tables.

Fill every `<...>` from the sweep JSON files in `bench_msgs/`:
`sweep_env_interact.json`, `sweep_rollout_predict.json`,
`sweep_actor_micro_batch.json`. **Cite specific fields for every claim** —
"step_ms=54,966 ± 425 at bs=96", not "env is slow".

---

## Template

```markdown
# Atomic Profile Report — `<config_name>`

> **Experiment:** `<experiment_name>`
> **Date:** YYYY-MM-DD
> **Config:** `<path_to_config_yaml>`
> **Model:** `<model_name>` (`<params>`, LoRA rank `<R>`, `<dtype>`)
> **Dataset:** `<dataset_name>` (train: `<N>` envs, eval: `<N>` envs)
> **Hardware:** `<N>` × `<GPU_name>` (`<mem>`), `<N>` CPU cores
> **GPU Layout:** Env (`<gpu_ids>`), Rollout (`<gpu_ids>`), Actor (`<gpu_ids>`, `<sharding>`)

---

## 1. Environment Sweep — `env.interact_step`

> Measures: bootstrap time, step latency, GPU VRAM, host RSS, CPU utilization.

| num_envs | factor | step_ms | bootstrap_ms | offload_ms | peak_alloc_MB | host_rss_MB | cpu%_mean | gpu%_mean | Notes |
|---------:|-------:|--------:|-------------:|-----------:|--------------:|------------:|----------:|----------:|:------|
| `<N>` | `<factor>` | `<mean> ± <std>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | `<%>` | |
| `<N>` | `<factor>` | `<mean> ± <std>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | `<%>` | |
| **`<N>`** | **1×** | **`<mean> ± <std>`** | **`<ms>`** | **`<ms>`** | **`<MB>`** | **`<MB>`** | **`<%>`** | **`<%>`** | **default** |
| `<N>` | `<factor>` | `<mean> ± <std>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | `<%>` | |
| `<N>` | `<factor>` | `<mean> ± <std>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | `<%>` | |

### Analysis

- **Scaling pattern:** <linear / sub-linear / super-linear — cite step_ms per unit>
- **GPU utilization:** <is the env GPU-bound or CPU-bound? cite gpu%_mean>
- **CPU utilization:** <multicore or single-core? cite cpu%_mean, cpu%_peak>
- **Host RSS scaling:** <RSS growth per N envs>
- **Bootstrap/Offload:** <stable or not>
- **Bottleneck:** <what limits further scaling>

**Conclusion:** `<recommended num_envs range with rationale>`

---

## 2. Rollout Sweep — `rollout.predict`

> Measures: inference latency, GPU VRAM, GPU utilization across batch sizes.

| batch_size | factor | ms_mean | ms_std | peak_alloc_MB | peak_reserved_MB | gpu%_mean | Notes |
|-----------:|-------:|--------:|-------:|--------------:|-----------------:|----------:|:------|
| `<N>` | `<factor>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |
| `<N>` | `<factor>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |
| **`<N>`** | **1×** | **`<ms>`** | **`<ms>`** | **`<MB>`** | **`<MB>`** | **`<%>`** | **default** |
| `<N>` | `<factor>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |
| `<N>` | `<factor>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |

| Weight op | Latency | peak_alloc_MB |
|:----------|--------:|--------------:|
| **Onload** (CPU→GPU) | `<ms>` | `<MB>` |
| **Offload** (GPU→CPU) | `<ms>` | `<MB>` |

### Analysis

- **Scaling pattern:** <linear / sub-linear / super-linear — cite ms per unit batch>
- **GPU utilization:** <saturated or not? cite gpu%_mean>
- **VRAM headroom:** <peak vs total, OOM boundary estimate>
- **Latency stability:** <std as % of mean>
- **Onload/Offload cost:** <acceptable or not>

**Conclusion:** `<recommended batch size range with rationale>`

---

## 3. Actor Training Sweep — `actor.train_micro_batch`

> Measures: training step latency, VRAM, GPU utilization sweeping micro_batch_size
> with fixed global_batch_size.

| micro_batch | num_micro | ms_mean | ms_std | peak_alloc_MB | peak_reserved_MB | gpu%_mean | Notes |
|------------:|----------:|--------:|-------:|--------------:|-----------------:|----------:|:------|
| `<N>` | `<N>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | `<N>` micro-steps |
| `<N>` | `<N>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |
| **`<N>`** | **`<N>`** | **`<ms>`** | **`<ms>`** | **`<MB>`** | **`<MB>`** | **`<%>`** | **default** |
| `<N>` | `<N>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |
| `<N>` | `<N>` | `<ms>` | `<ms>` | `<MB>` | `<MB>` | `<%>` | |
| `<N>` | 1 | **OOM** | — | — | — | — | single micro-step OOM |

### Analysis

- **Micro-step overhead:** <smaller micro-batches → more micro-steps → more sync overhead>
- **Memory–speed trade-off:** <quantify the trade-off: larger micro-batch saves time but costs VRAM>
- **OOM boundary:** <what micro_batch_size OOMs and why>
- **GPU utilization:** <how util varies with micro_batch_size>
- **Sweet spot:** <recommended micro_batch_size with rationale>

**Conclusion:** `<recommended micro_batch_size with rationale>`

---

## 4. Summary & Recommendations

### Key Metrics

| Phase | Default Config | Latency | Per-GPU VRAM | Bottleneck |
|:------|:--------------|--------:|:------------:|:----------|
| Env | `<num_envs>` envs/GPU | `<s>` | `<GB>` | `<CPU/GPU>` |
| Rollout | batch `<N>` | `<s>` | `<GB>` | `<bottleneck>` |
| Actor | micro=`<N>` × `<N>` | `<s>` | `<GB>` | `<bottleneck>` |

### End-to-End Step Simulation

```
1. Env interact (<N> envs × <N> GPUs = <N> envs/GPU):  <s>
2. Rollout predict (batch <N> × <N> GPUs):             <s>
3. Actor train (micro=<N> × <N> × <N> GPUs):           <s>
-----------------------------------------------------
   Total step time (parallel phases → max):            ~<s>
```

### Optimization Recommendations

| Component | Suggestion | Expected Benefit |
|:----------|:-----------|:-----------------|
| **Env** | `<suggestion>` | `<expected throughput gain>` |
| **Rollout** | `<suggestion>` | `<expected throughput gain>` |
| **Actor** | `<suggestion>` | `<expected throughput gain>` |
| **Actor** | `<suggestion>` | `<expected throughput gain>` |

### Resource Utilization Summary

| Resource | Peak Used | Available | Utilization |
|:---------|:---------|:---------|:-----------:|
| GPU VRAM (rollout) | `<GB>` | `<GB>` | `<%>` |
| GPU VRAM (actor) | `<GB>` | `<GB>` | `<%>` |
| GPU VRAM (env) | `<GB>` | `<GB>` | `<%>` |
| Host RAM (env) | `<GB>` | `<GB>` | `<%>` |

---

*Report generated by RLinf Atomic Profiler (`benchmark_sweep.py`)*
```

---

## Style rules

- **Per-component, isolated.** Each sweep measures one component on disjoint GPUs
  — never mix env/rollout/actor measurements in the same table.
- **Cite the field.** Every claim names the JSON field behind it (e.g.
  `ms_mean`, `peak_alloc_MB`, `gpu_util_mean`).
- **Show the default.** Always highlight the default config row (bold) so the
  reader sees the baseline before considering changes.
- **Rank by bottleneck impact.** Fix the component that dominates total step time
  first, then the next, then the next.
- **One table per component.** Three separate tables (env, rollout, actor) each
  with its own analysis section — don't collapse them.
- **Quantify linearity.** State whether the component scales linearly, sub-linearly,
  or super-linearly with its knob, and cite the per-unit cost.
- **End-to-end simulation.** Always include the step simulation section to show
  how the three components compose into total step time.
- **OOM boundary is a data point.** If a sweep point OOMs, record it in the table
  — that's as valuable as any successful measurement.
- **Link raw data.** Reference the JSON/CSV files in `bench_msgs/` for readers
  who want to verify or re-plot.

## Anti-patterns

- ❌ Mixing measurements from different components in the same table.
- ❌ Recommending knob values without citing the sweep curve.
- ❌ Omitting the OOM boundary — the reader needs to know where the ceiling is.
- ❌ Only reporting latency, not VRAM — both are needed for production configs.
- ❌ No end-to-end simulation — the per-component numbers don't compose obviously.
- ❌ Ignoring the onload/offload cost — it's part of the wall time in production.
- ❌ Claiming "linear" without citing the per-unit cost or R² fit.