#!/usr/bin/env python3
"""Diagnose a single RLinf embodiment profiling run.

Reads a run's <log_dir> (timeline JSONL + nvitop summary + metrics.log +
config snapshot) and emits a structured diagnosis.json (+ human diagnosis.txt)
of the signals the diagnosis playbook keys on. It extracts signals only — it
does not decide fixes; reference/diagnosis-playbook.md maps signals -> fixes.

Usage:
    python diagnose.py <log_dir> [-o <log_dir>/diagnosis.json]

Stdlib-only so it runs under any venv without extra installs.

Key domain facts (see reference/concepts.md):
  * Metric of merit is per-trajectory time = Step Time / num_trajectories,
    NOT absolute Step Time.
  * `generate_rollouts` (rollout/generate) is a PARENT span: rollout `predict`
    and env `env_interact_step` nest inside it. The top-level pipeline split is
    computed nesting-aware so children are never summed into the top level.
  * A run with nvitop samples but no metrics.log likely OOM'd before step 1.
  * Warmup: the first 2 occurrences per rank of a repeated tag are dropped
    before averaging (mirrors compute_timeline_avg.py).
"""
import argparse
import glob
import json
import os
import re
from statistics import mean

SKIP_WARMUP = 2  # drop first N occurrences per rank when averaging repeated tags

# --- tag groups for the top-level pipeline split (alias-tolerant) ------------
GENERATION = {"generate_rollouts", "rollout/generate", "generate", "generate_one_epoch",
              "rollout/generate_one_epoch"}
# There is usually no single `run_training` span in the timeline; training is the
# union of the actor compute micro-ops (plus the wrapper tag if a build emits one).
TRAINING = {"actor_forward", "actor_backward", "actor_policy_loss",
            "actor_optimizer_step", "actor/run_training", "run_training"}
WEIGHT_SYNC = {"actor/sync_model_to_rollout", "sync_model_to_rollout", "sync_weights",
               "env/env/send_rollout_trajectories", "send_rollout_trajectories"}
ADV = {"actor/compute_adv", "compute_adv", "cal_adv_and_returns"}
ACTOR_IDLE = {"actor/recv_traj", "recv_traj"}  # actor blocked waiting on rollout
# repeated inner tags for straggler analysis
HOT_TAGS = ["predict", "env_interact_step", "actor_forward", "actor_backward",
            "actor_optimizer_step"]


def _is_offload(tag):
    t = tag.lower()
    return "offload" in t or t.startswith("load_") or "reload" in t


# --- generic interval helpers ------------------------------------------------
def merge_intervals(intervals):
    """Union duration (s) of a list of (t0, t1)."""
    if not intervals:
        return 0.0
    ivs = sorted(intervals)
    total = 0.0
    cur0, cur1 = ivs[0]
    for a, b in ivs[1:]:
        if a > cur1:
            total += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    total += cur1 - cur0
    return total


# --- loaders -----------------------------------------------------------------
def load_timeline(log_dir):
    """Return list of event dicts from timeline/<comp>_rank<N>.jsonl."""
    events = []
    tdir = os.path.join(log_dir, "timeline")
    for path in sorted(glob.glob(os.path.join(tdir, "*.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "t0" in r and "t1" in r and "tag" in r:
                events.append(r)
    return events


def parse_metrics(log_dir):
    """Extract Step Time (s), num_trajectories, global step from metrics.log."""
    path = os.path.join(log_dir, "metrics.log")
    if not os.path.exists(path):
        return None
    text = open(path, errors="replace").read()
    out = {}
    m = re.search(r"Step Time:\s*([\d.]+)\s*s", text)
    if m:
        out["step_time_s"] = float(m.group(1))
    m = re.search(r"num_trajectories=([\d.]+)", text)
    if m:
        out["num_trajectories"] = int(float(m.group(1)))
    m = re.search(r"Global Step:\s*([\d]+)\s*/\s*([\d]+)", text)
    if m:
        out["global_step"] = int(m.group(1))
        out["total_steps"] = int(m.group(2))
    # a few env/quality signals when present
    for key in ("success_once", "return", "episode_len", "reward"):
        mm = re.search(rf"\b{re.escape(key)}=([\-\d.eE]+)", text)
        if mm:
            out[key] = float(mm.group(1))
    return out or None


def parse_config(log_dir):
    """Config knobs: config_snapshot.json -> run_embodiment.log CMD -> gap."""
    snap = os.path.join(log_dir, "config_snapshot.json")
    if os.path.exists(snap):
        try:
            data = json.load(open(snap))
            return {"source": "config_snapshot.json",
                    "name": data.get("name"),
                    "key_params": data.get("key_params", {}),
                    "overrides": data.get("overrides", [])}
        except (json.JSONDecodeError, OSError):
            pass
    runlog = os.path.join(log_dir, "run_embodiment.log")
    if os.path.exists(runlog):
        first = open(runlog, errors="replace").readline().strip()
        cfg = {}
        m = re.search(r"--config-name\s+(\S+)", first)
        if m:
            cfg["name"] = m.group(1)
        overrides = re.findall(r"(\S+=\S+)", first)
        # keep only dotted hydra-style overrides, drop the log_path noise
        cfg["overrides"] = [o for o in overrides if "." in o.split("=")[0]
                            and not o.startswith("runner.logger.log_path")]
        return {"source": "run_embodiment.log", **cfg}
    return {"source": None}


def parse_nvitop_summary(log_dir):
    """Parse nvitop/nvitop_summary.log into gpus / overall / processes / meta."""
    path = os.path.join(log_dir, "nvitop", "nvitop_summary.log")
    if not os.path.exists(path):
        return None
    out = {"gpus": {}, "overall": {}, "processes": {}, "meta": {}}

    def kv(line):
        return {k: float(v) for k, v in re.findall(r"(\w+)=([\-\d.]+)", line)}

    for line in open(path, errors="replace"):
        s = line.strip()
        m = re.match(r"^(gpu\d+):\s*(.*)", s)
        if m:
            out["gpus"][m.group(1)] = kv(m.group(2))
            continue
        m = re.match(r"^(overall_\w+):\s*(.*)", s)
        if m:
            out["overall"][m.group(1)] = kv(m.group(2))
            continue
        m = re.match(r"^(\w+)/r(\d+)/pid(\d+):\s*(.*)", s)
        if m:
            comp, rank, pid, rest = m.groups()
            rec = kv(rest)
            gi = re.search(r"gpu_indices=(\[[^\]]*\]|n/a)", rest)
            rec["gpu_indices"] = gi.group(1) if gi else None
            out["processes"][f"{comp}/r{rank}"] = {"component": comp, "rank": int(rank),
                                                   "pid": int(pid), **rec}
            continue
        m = re.match(r"^(samples|span_s|aggregate_bin_s):\s*([\d.]+)", s)
        if m:
            out["meta"][m.group(1)] = float(m.group(2))
    return out


def gpu_mem_total_gib(log_dir):
    """Read one nvitop jsonl sample to learn per-GPU total memory (GiB)."""
    for path in sorted(glob.glob(os.path.join(log_dir, "nvitop", "*.jsonl"))):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            for g in r.get("gpus", []):
                if g.get("memory_total_gib"):
                    return float(g["memory_total_gib"])
        break
    return None


# --- analyses ----------------------------------------------------------------
def component_of(ev):
    return ev.get("component", "?")


def stage_breakdown(events):
    """Nesting-aware per-(component,tag) aggregation.

    Depth is assigned per (component, rank) by interval containment. Top-level
    entries are depth==0; deeper entries are children (e.g. predict under
    generate). Durations reported are summed occurrences and merged wall-clock.
    """
    # group events by (component, rank)
    groups = {}
    for ev in events:
        groups.setdefault((component_of(ev), ev.get("rank", 0)), []).append(ev)

    agg = {}  # (component, tag, depth) -> {"sum","max","count","durs"}
    for (_comp, _rank), evs in groups.items():
        evs = sorted(evs, key=lambda e: (e["t0"], -e["t1"]))
        stack = []  # open intervals (t1)
        for ev in evs:
            t0, t1 = ev["t0"], ev["t1"]
            while stack and stack[-1] < t0:
                stack.pop()
            depth = len(stack)
            stack.append(t1)
            key = (component_of(ev), ev["tag"], depth)
            d = t1 - t0
            slot = agg.setdefault(key, {"sum": 0.0, "max": 0.0, "count": 0})
            slot["sum"] += d
            slot["max"] = max(slot["max"], d)
            slot["count"] += 1
    # serialize
    rows = []
    for (comp, tag, depth), slot in sorted(agg.items(), key=lambda x: -x[1]["sum"]):
        rows.append({"component": comp, "tag": tag, "depth": depth,
                     "total_s": round(slot["sum"], 3), "max_s": round(slot["max"], 3),
                     "count": slot["count"],
                     "mean_s": round(slot["sum"] / slot["count"], 4)})
    return rows


def tag_merged_duration(events, tag_set):
    ivs = [(e["t0"], e["t1"]) for e in events if e["tag"] in tag_set]
    return merge_intervals(ivs)


def pipeline_split(events, span_s, denom_s=None, denom_src="timeline_span"):
    """Top-level phase split. Durations are merged wall-clock (nesting-aware:
    children like predict/env_interact are never summed into the top level).
    Percentages use denom_s (prefer metrics Step Time) so they match the
    per-step story rather than being diluted by setup/warmup in the full span.
    """
    if span_s <= 0:
        return {}
    denom = denom_s if denom_s and denom_s > 0 else span_s
    gen = tag_merged_duration(events, GENERATION)
    train = tag_merged_duration(events, TRAINING)
    sync = tag_merged_duration(events, WEIGHT_SYNC)
    adv = tag_merged_duration(events, ADV)
    idle = tag_merged_duration(events, ACTOR_IDLE)
    offload = merge_intervals([(e["t0"], e["t1"]) for e in events if _is_offload(e["tag"])])

    def pct(x):
        return round(100.0 * x / denom, 1)

    return {
        "span_s": round(span_s, 3),
        "denominator_s": round(denom, 3), "denominator_src": denom_src,
        "generation_s": round(gen, 3), "generation_pct": pct(gen),
        "training_s": round(train, 3), "training_pct": pct(train),
        "weight_sync_s": round(sync, 3), "weight_sync_pct": pct(sync),
        "adv_s": round(adv, 3), "adv_pct": pct(adv),
        "actor_idle_during_generation_s": round(idle, 3),
        "actor_idle_during_generation_pct": pct(idle),
        "offload_cost_s": round(offload, 3), "offload_cost_pct": pct(offload),
    }


def generation_internals(events):
    """Within the generation phase: predict vs env_interact busy time."""
    predict = tag_merged_duration(events, {"predict"})
    env_step = tag_merged_duration(events, {"env_interact_step"})
    gen = tag_merged_duration(events, GENERATION)
    out = {"generation_s": round(gen, 3),
           "predict_busy_s": round(predict, 3),
           "env_interact_busy_s": round(env_step, 3)}
    if gen > 0:
        out["predict_pct_of_generation"] = round(100.0 * predict / gen, 1)
        out["env_interact_pct_of_generation"] = round(100.0 * env_step / gen, 1)
    return out


def rank_straggler(events):
    """max/mean per-rank mean duration for hot repeated tags (warmup dropped)."""
    # (tag, rank) -> [durations sorted by t0]
    per = {}
    for ev in events:
        if ev["tag"] in HOT_TAGS:
            per.setdefault((ev["tag"], ev.get("rank", 0)), []).append((ev["t0"],
                                                                       ev["t1"] - ev["t0"]))
    out = {}
    tags = {}
    for (tag, rank), lst in per.items():
        lst.sort()
        durs = [d for _, d in lst[SKIP_WARMUP:]]
        if durs:
            tags.setdefault(tag, {})[rank] = mean(durs)
    for tag, rankmeans in tags.items():
        vals = list(rankmeans.values())
        mx, mn = max(vals), mean(vals)
        slowest = max(rankmeans, key=rankmeans.get)
        out[tag] = {"per_rank_mean_s": {str(k): round(v, 4) for k, v in sorted(rankmeans.items())},
                    "mean_s": round(mn, 4), "max_s": round(mx, 4),
                    "straggler_ratio": round(mx / mn, 3) if mn else None,
                    "slowest_rank": slowest}
    return out


def offload_cost_breakdown(events):
    """Per-component per-rank onload and offload time (warmup dropped).

    Dynamically detects onload / offload tags per component:
      - offload: tag contains "offload" (e.g. offload_model, offload, offload_param_and_grad)
      - onload:  tag contains "load" but NOT "offload"
                 (e.g. reload_model, onload, load_param_and_grad, load_optimizer)
    """
    if not events:
        return {}
    # (component, direction, rank) -> [(t0, duration)]
    per = {}
    for ev in events:
        tag = ev["tag"]
        t = tag.lower()
        if "offload" in t:
            direction = "offload"
        elif "load" in t:
            direction = "onload"
        else:
            continue
        comp = ev.get("component", "?")
        rank = ev.get("rank", 0)
        per.setdefault((comp, direction, rank), []).append((ev["t0"], ev["t1"] - ev["t0"]))

    # per-rank summed durations (warmup dropped when enough samples; offload/onload
    # are low-frequency — a run may have only 2–4 events per rank per direction)
    rank_sums = {}  # (comp, direction) -> {rank: sum_s}
    gpu_counts = {}  # comp -> unique rank count (≈ GPU count)
    for (comp, direction, rank), lst in per.items():
        lst.sort()
        durs = [d for _, d in lst]
        if len(durs) > SKIP_WARMUP:
            durs = durs[SKIP_WARMUP:]
        if durs:
            rank_sums.setdefault((comp, direction), {})[rank] = sum(durs)
        gpu_counts[comp] = max(gpu_counts.get(comp, 0), rank + 1)

    # aggregate per component × direction (stats across per-rank summed durations)
    out = {}
    for (comp, direction), rankvals in sorted(rank_sums.items()):
        vals = list(rankvals.values())
        mn = mean(vals)
        mx = max(vals)
        slowest = max(rankvals, key=rankvals.get)
        ngpu = gpu_counts.get(comp, len(vals))
        out.setdefault(comp, {})[direction] = {
            "per_rank_s": {str(k): round(v, 4) for k, v in sorted(rankvals.items())},
            "mean_s": round(mn, 4), "max_s": round(mx, 4),
            "straggler_ratio": round(mx / mn, 3) if mn else None,
            "slowest_rank": slowest,
            "mean_per_gpu_s": round(mn / ngpu, 4) if ngpu else None,
        }
    return out


def gpu_and_memory(nv, total_gib):
    if not nv:
        return None, None
    util = {"per_gpu": {}, "overall": nv.get("overall", {}), "per_process": {}}
    for g, rec in nv["gpus"].items():
        util["per_gpu"][g] = {k: rec.get(k) for k in
                              ("avg_gpu_util", "max_gpu_util", "avg_mem", "max_mem")}
    for name, rec in nv.get("processes", {}).items():
        util["per_process"][name] = {
            "component": rec.get("component"), "rank": rec.get("rank"),
            "avg_gpu_util": rec.get("avg_process_gpu_util"),
            "max_gpu_util": rec.get("max_process_gpu_util"),
            "avg_gpu_mem_gib": rec.get("avg_process_gpu_mem"),
            "max_gpu_mem_gib": rec.get("max_process_gpu_mem"),
            "gpu_indices": rec.get("gpu_indices"),
        }
    # idle-group heuristic: GPUs whose avg util is < 0.7 * the busiest gpu's avg
    avgs = {g: rec.get("avg_gpu_util", 0.0) for g, rec in nv["gpus"].items()}
    if avgs:
        busiest = max(avgs.values()) or 1.0
        util["low_util_gpus"] = sorted(g for g, v in avgs.items() if v < 0.7 * busiest)

    max_mem = max((rec.get("max_mem", 0.0) for rec in nv["gpus"].values()), default=0.0)
    mem = {"max_used_gib": round(max_mem, 2), "total_gib": total_gib}
    if total_gib:
        mem["max_used_pct"] = round(100.0 * max_mem / total_gib, 1)
        mem["headroom_gib"] = round(total_gib - max_mem, 2)
        mem["oom_risk"] = ("high" if (total_gib - max_mem) < 1.0 else
                           "elevated" if (total_gib - max_mem) < 5.0 else "safe")
    return util, mem


def cpu_saturation(nv):
    if not nv:
        return None
    out = {}
    for name, rec in nv["processes"].items():
        out[name] = {"avg_cpu": rec.get("avg_cpu"), "max_cpu": rec.get("max_cpu"),
                     "gpu_indices": rec.get("gpu_indices")}
    # flag processes pinning many cores (>200% = >2 cores)
    out["_multicore_flagged"] = sorted(
        n for n, r in out.items() if isinstance(r, dict)
        and (r.get("max_cpu") or 0) > 200.0)
    return out


# --- main --------------------------------------------------------------------
def diagnose(log_dir):
    events = load_timeline(log_dir)
    metrics = parse_metrics(log_dir)
    config = parse_config(log_dir)
    nv = parse_nvitop_summary(log_dir)
    total_gib = gpu_mem_total_gib(log_dir)

    gaps = []
    if not events:
        gaps.append("timeline/: no events found")
    if metrics is None:
        gaps.append("metrics.log: missing")
    if nv is None:
        gaps.append("nvitop/nvitop_summary.log: missing (run plot_nvitop first)")
    if config.get("source") is None:
        gaps.append("config: no config_snapshot.json or run_embodiment.log")

    span_s = 0.0
    components, ranks = set(), set()
    if events:
        span_s = max(e["t1"] for e in events) - min(e["t0"] for e in events)
        for e in events:
            components.add(component_of(e))
            ranks.add((component_of(e), e.get("rank", 0)))

    # efficiency headline
    eff = {}
    if metrics:
        eff.update({k: metrics[k] for k in
                    ("step_time_s", "num_trajectories", "global_step", "total_steps")
                    if k in metrics})
        if metrics.get("step_time_s") and metrics.get("num_trajectories"):
            eff["step_time_per_traj_s"] = round(
                metrics["step_time_s"] / metrics["num_trajectories"], 4)
        for k in ("success_once", "return", "episode_len", "reward"):
            if k in metrics:
                eff[k] = metrics[k]
    util, mem = gpu_and_memory(nv, total_gib)

    # prefer metrics Step Time as the pipeline-split denominator (matches the
    # per-step story); fall back to the full timeline wall-clock span.
    step_t = (metrics or {}).get("step_time_s")
    denom_s = step_t if step_t else span_s
    denom_src = "metrics.step_time_s" if step_t else "timeline_span"

    result = {
        "meta": {
            "log_dir": os.path.abspath(log_dir),
            "config": config,
            "components": sorted(components),
            "num_ranks": len(ranks),
            "nvitop_span_s": (nv or {}).get("meta", {}).get("span_s") if nv else None,
            "timeline_span_s": round(span_s, 3),
        },
        "efficiency": eff,
        "pipeline_split": pipeline_split(events, span_s, denom_s, denom_src) if events else {},
        "offload_cost": offload_cost_breakdown(events) if events else {},
        "generation_internals": generation_internals(events) if events else {},
        "rank_straggler": rank_straggler(events) if events else {},
        "gpu_utilization": util,
        "memory": mem,
        "cpu_saturation": cpu_saturation(nv),
        "stage_breakdown": stage_breakdown(events) if events else [],
        "data_gaps": gaps,
    }
    return result


def to_text(d):
    """Compact human-readable summary."""
    L = []
    m = d["meta"]
    L.append(f"# Diagnosis: {m['log_dir']}")
    cfg = m["config"]
    if cfg.get("key_params"):
        L.append("config: " + ", ".join(f"{k}={v}" for k, v in cfg["key_params"].items()))
    elif cfg.get("overrides"):
        L.append("config overrides: " + " ".join(cfg["overrides"]))
    L.append(f"components: {', '.join(m['components'])}  ranks: {m['num_ranks']}  "
             f"span: {m['timeline_span_s']}s")
    e = d["efficiency"]
    if "step_time_per_traj_s" in e:
        L.append(f"\n## Efficiency (metric of merit)\n"
                 f"step_time={e.get('step_time_s')}s  num_traj={e.get('num_trajectories')}  "
                 f"=> per-traj={e['step_time_per_traj_s']}s")
    if e.get("likely_oom_before_first_step"):
        L.append("!! nvitop present but no metrics.log => LIKELY OOM before first step")
    ps = d["pipeline_split"]
    if ps:
        L.append(f"\n## Pipeline split (denom {ps['denominator_s']}s "
                 f"[{ps['denominator_src']}], full span {ps['span_s']}s)\n"
                 f"generation={ps['generation_pct']}%  training={ps['training_pct']}%  "
                 f"weight_sync={ps['weight_sync_pct']}%  adv={ps['adv_pct']}%\n"
                 f"actor_idle_during_generation={ps['actor_idle_during_generation_pct']}%  "
                 f"offload_cost={ps['offload_cost_pct']}%")
    oc = d.get("offload_cost", {})
    if oc:
        L.append("\n## Offload cost per-component (per-rank sum, warmup dropped)")
        for comp in sorted(oc):
            for direction in ("offload", "onload"):
                info = oc[comp].get(direction)
                if not info:
                    continue
                gpu_s = info.get("mean_per_gpu_s")
                gpu_str = f" per_gpu={gpu_s}s" if gpu_s is not None else ""
                L.append(f"  {comp}/{direction}: mean={info['mean_s']}s max={info['max_s']}s "
                         f"ratio={info['straggler_ratio']} slowest=r{info['slowest_rank']}{gpu_str}")
                for rank, val in sorted(info.get("per_rank_s", {}).items(),
                                        key=lambda x: int(x[0])):
                    L.append(f"    r{rank}: {val}s")
    gi = d["generation_internals"]
    if gi.get("generation_s"):
        L.append(f"generation internals: predict_busy={gi.get('predict_busy_s')}s "
                 f"({gi.get('predict_pct_of_generation')}%), "
                 f"env_interact_busy={gi.get('env_interact_busy_s')}s "
                 f"({gi.get('env_interact_pct_of_generation')}%)")
    rs = d["rank_straggler"]
    if rs:
        L.append("\n## Rank straggler (max/mean, warmup dropped)")
        for tag, r in rs.items():
            L.append(f"  {tag}: ratio={r['straggler_ratio']} "
                     f"(mean={r['mean_s']}s max={r['max_s']}s slowest=r{r['slowest_rank']})")
    if d["gpu_utilization"]:
        ov = d["gpu_utilization"]["overall"].get("overall_active_gpus", {})
        L.append(f"\n## GPU\navg_util={ov.get('avg_gpu_util')}%  "
                 f"low_util_gpus={d['gpu_utilization'].get('low_util_gpus')}")
        pp = d["gpu_utilization"].get("per_process", {})
        if pp:
            L.append("  per-process GPU util:")
            for name, rec in sorted(pp.items()):
                avg_u = rec.get("avg_gpu_util")
                max_u = rec.get("max_gpu_util")
                mem = rec.get("avg_gpu_mem_gib")
                gpus = rec.get("gpu_indices")
                parts = [f"avg_gpu={avg_u}%" if avg_u is not None else "",
                         f"max_gpu={max_u}%" if max_u is not None else "",
                         f"avg_mem={mem} GiB" if mem is not None else "",
                         f"gpus={gpus}" if gpus else ""]
                L.append(f"    {name}: " + " ".join(p for p in parts if p))
    if d["memory"]:
        mm = d["memory"]
        L.append(f"memory: max_used={mm.get('max_used_gib')} GiB "
                 f"({mm.get('max_used_pct')}% of {mm.get('total_gib')})  "
                 f"oom_risk={mm.get('oom_risk')}")
    cs = d["cpu_saturation"]
    if cs and cs.get("_multicore_flagged"):
        L.append(f"cpu multicore (>2 cores): {', '.join(cs['_multicore_flagged'])}")
    if d["data_gaps"]:
        L.append("\n## Data gaps\n  - " + "\n  - ".join(d["data_gaps"]))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Diagnose one RLinf embodiment run.")
    ap.add_argument("log_dir", help="run log dir (contains timeline/, nvitop/, metrics.log)")
    ap.add_argument("-o", "--output", default=None,
                    help="diagnosis.json path (default: <log_dir>/diagnosis.json)")
    ap.add_argument("--text-output", default=None,
                    help="diagnosis.txt path (default: <log_dir>/diagnosis.txt)")
    args = ap.parse_args()

    d = diagnose(args.log_dir)
    out = args.output or os.path.join(args.log_dir, "diagnosis.json")
    txt = args.text_output or os.path.join(args.log_dir, "diagnosis.txt")
    with open(out, "w") as f:
        json.dump(d, f, indent=2)
    with open(txt, "w") as f:
        f.write(to_text(d))
    print(to_text(d))
    print(f"\nwrote {out}\nwrote {txt}")


if __name__ == "__main__":
    main()
