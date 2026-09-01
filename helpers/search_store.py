#!/usr/bin/env python3
"""Node store + tree for the embodiment config search (beam search).

Maintains, per campaign directory:
  * baseline_knobs.json — baseline knob dict loaded from config YAML at init time
  * nodes.jsonl  — append-only log of nodes (source of truth, resumable)
  * tree.json    — materialized view: each node with explicit parent + children[]

Each node maps: id -> config (overrides delta + resolved knobs) -> log_dir, plus
parent/children tree links, objective (step_time_per_traj_s; lower is better),
and status. Deterministic bookkeeping only — proposals are made by the caller
(the skill's LLM step). Stdlib+yaml.

CLI:
    search_store.py init      --campaign-dir D --log-dir L --config C [--overrides '{}'] [--tag baseline]
    search_store.py add       --campaign-dir D --parent P --overrides '<json>' --log-dir L [--round R] [--tag T]
    search_store.py set-result --campaign-dir D --id I (--from-diagnosis F | --objective X [--status S] [--failure M])
    search_store.py dedup     --campaign-dir D --parent P --overrides '<json>'
    search_store.py frontier  --campaign-dir D [--k 2] [--composite|--no-composite] [--alpha A] [--beta B] [--gamma G]
    search_store.py tree      --campaign-dir D
    search_store.py best      --campaign-dir D
    search_store.py exhaustion-check --campaign-dir D [--k 2] [--min-headroom-gib 5.0] [--min-rounds 5]

Config resolution + hashing reuse preflight.resolve so the SHA matches what
preflight validates.  The baseline is read from a config YAML at init time and
persisted as baseline_knobs.json so subsequent commands don't need the YAML.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight  # sibling helper: resolve(), load_baseline_from_config()

NODES_FILE = "nodes.jsonl"
TREE_FILE = "tree.json"
BASELINE_FILE = "baseline_knobs.json"

# statuses / failure modes
ST_ROOT, ST_OK, ST_FAILED, ST_DUPLICATE = "ROOT", "OK", "FAILED", "DUPLICATE"


# --- baseline persistence ----------------------------------------------------

def _load_baseline(campaign_dir):
    """Load baseline knobs from the campaign dir, falling back to defaults."""
    path = os.path.join(campaign_dir, BASELINE_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return dict(preflight._DEFAULT_KNOB_VALUES)


# --- low-level store ---------------------------------------------------------

def _nodes_path(campaign_dir):
    return os.path.join(campaign_dir, NODES_FILE)


def load_nodes(campaign_dir):
    """Return list of node dicts (corruption-tolerant: skip bad lines)."""
    path = _nodes_path(campaign_dir)
    nodes = []
    if not os.path.exists(path):
        return nodes
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            nodes.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return nodes


def _append_node(campaign_dir, node):
    path = _nodes_path(campaign_dir)
    with open(path, "a") as f:
        f.write(json.dumps(node) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _by_id(nodes):
    return {n["id"]: n for n in nodes}


def _children_map(nodes):
    ch = {n["id"]: [] for n in nodes}
    for n in nodes:
        p = n.get("parent")
        if p is not None and p in ch:
            ch[p].append(n["id"])
    for v in ch.values():
        v.sort()
    return ch


def _resolve_for(campaign_dir, nodes_by_id, parent_id, overrides):
    """Cumulative resolved knobs = baseline + chain of ancestor overrides + this delta."""
    baseline = _load_baseline(campaign_dir)
    # walk parent chain root->this, applying overrides in order
    chain = []
    cur = parent_id
    while cur is not None:
        n = nodes_by_id.get(cur)
        if n is None:
            break
        chain.append(n.get("overrides") or {})
        cur = n.get("parent")
    chain.reverse()
    resolved = dict(baseline)
    for delta in chain:
        resolved.update(delta)
    resolved.update(overrides or {})
    return resolved


def _sha(resolved):
    payload = json.dumps({k: resolved[k] for k in sorted(resolved)}, sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()[:12]


def _write_tree(campaign_dir, nodes):
    nodes = _latest(nodes)  # collapse append-only log (latest record per id wins)
    ch = _children_map(nodes)
    view = {"nodes": [], "leaderboard": []}
    for n in sorted(nodes, key=lambda x: x["id"]):
        view["nodes"].append({
            "id": n["id"], "parent": n.get("parent"), "children": ch[n["id"]],
            "tag": n.get("tag"), "status": n.get("status"),
            "objective": n.get("objective"), "failure_mode": n.get("failure_mode"),
            "overrides": n.get("overrides") or {}, "log_dir": n.get("log_dir"),
            "config_sha": n.get("config_sha"), "round": n.get("round"),
            "headroom_gib": n.get("headroom_gib"),
            "success_once": n.get("success_once"),
        })
    ok = [n for n in nodes if n.get("status") == ST_OK and n.get("objective") is not None]
    for n in sorted(ok, key=lambda x: x["objective"]):
        view["leaderboard"].append({"id": n["id"], "tag": n.get("tag"),
                                    "objective": n["objective"],
                                    "headroom_gib": n.get("headroom_gib"),
                                    "success_once": n.get("success_once")})
    with open(os.path.join(campaign_dir, TREE_FILE), "w") as f:
        json.dump(view, f, indent=2)


# --- commands ----------------------------------------------------------------

def cmd_init(a):
    os.makedirs(a.campaign_dir, exist_ok=True)
    if load_nodes(a.campaign_dir):
        raise SystemExit(f"campaign already initialized: {_nodes_path(a.campaign_dir)}")

    # load baseline from config YAML
    config_path = a.config
    if config_path:
        baseline = preflight.load_baseline_from_config(config_path)
    else:
        baseline = dict(preflight._DEFAULT_KNOB_VALUES)

    # persist baseline
    with open(os.path.join(a.campaign_dir, BASELINE_FILE), "w") as f:
        json.dump(baseline, f, indent=2)

    overrides = json.loads(a.overrides) if a.overrides else {}
    resolved = dict(baseline)
    resolved.update(overrides or {})

    node = {"id": 0, "parent": None, "overrides": overrides,
            "resolved_config": resolved, "config_sha": _sha(resolved),
            "log_dir": os.path.abspath(a.log_dir), "objective": None,
            "status": ST_ROOT, "failure_mode": None, "round": 0,
            "tag": a.tag or "baseline"}
    _append_node(a.campaign_dir, node)
    _write_tree(a.campaign_dir, load_nodes(a.campaign_dir))
    print(json.dumps({"id": 0, "config_sha": node["config_sha"],
                      "baseline_knobs": baseline}))


def cmd_add(a):
    nodes = load_nodes(a.campaign_dir)
    if not nodes:
        raise SystemExit("campaign not initialized; run `init` first")
    by_id = _by_id(nodes)
    if a.parent not in by_id:
        raise SystemExit(f"unknown parent id: {a.parent}")
    overrides = json.loads(a.overrides)
    resolved = _resolve_for(a.campaign_dir, by_id, a.parent, overrides)
    sha = _sha(resolved)
    new_id = max(n["id"] for n in nodes) + 1
    node = {"id": new_id, "parent": a.parent, "overrides": overrides,
            "resolved_config": resolved, "config_sha": sha,
            "log_dir": os.path.abspath(a.log_dir) if a.log_dir else None,
            "objective": None, "status": a.status or ST_OK,
            "failure_mode": a.failure, "round": a.round, "tag": a.tag}
    _append_node(a.campaign_dir, node)
    _write_tree(a.campaign_dir, load_nodes(a.campaign_dir))
    print(json.dumps({"id": new_id, "config_sha": sha}))


def cmd_set_result(a):
    nodes = load_nodes(a.campaign_dir)
    by_id = _by_id(nodes)
    if a.id not in by_id:
        raise SystemExit(f"unknown id: {a.id}")
    objective, status, failure = a.objective, a.status, a.failure
    headroom_gib = None
    success_once = None
    if a.from_diagnosis:
        d = json.load(open(a.from_diagnosis))
        eff = d.get("efficiency", {})
        objective = eff.get("step_time_per_traj_s")
        # extract extra fields for composite scoring (A4)
        mem = d.get("memory", {})
        headroom_gib = mem.get("headroom_gib")
        success_once = eff.get("success_once")
        if eff.get("likely_oom_before_first_step"):
            status, failure = ST_FAILED, "OOM"
        elif objective is None:
            status, failure = ST_FAILED, "METRICS_MISSING"
        else:
            status = ST_OK
    node = dict(by_id[a.id])
    if objective is not None:
        node["objective"] = objective
    if status:
        node["status"] = status
    node["failure_mode"] = failure
    if headroom_gib is not None:
        node["headroom_gib"] = headroom_gib
    if success_once is not None:
        node["success_once"] = success_once
    _append_node(a.campaign_dir, node)
    _write_tree(a.campaign_dir, _latest(load_nodes(a.campaign_dir)))
    print(json.dumps({"id": a.id, "objective": node["objective"],
                      "status": node["status"], "failure_mode": node["failure_mode"],
                      "headroom_gib": node.get("headroom_gib"),
                      "success_once": node.get("success_once")}))


def _latest(nodes):
    """Collapse append-only log so the last record per id wins."""
    out = {}
    for n in nodes:
        out[n["id"]] = n
    return [out[k] for k in sorted(out)]


def cmd_dedup(a):
    nodes = _latest(load_nodes(a.campaign_dir))
    by_id = _by_id(nodes)
    overrides = json.loads(a.overrides)
    resolved = _resolve_for(a.campaign_dir, by_id, a.parent, overrides)
    sha = _sha(resolved)
    for n in nodes:
        if n.get("config_sha") == sha:
            print(json.dumps({"duplicate": True, "of_id": n["id"],
                              "config_sha": sha, "objective": n.get("objective"),
                              "status": n.get("status")}))
            return
    print(json.dumps({"duplicate": False, "config_sha": sha}))


def cmd_frontier(a):
    """Best-K OK nodes by composite or raw objective (A4 composite scoring).

    Composite scoring combines:
      - potential gain from memory headroom (α): nodes with room to grow get a bonus
      - exploration bonus (β): nodes with fewer children get a bonus (UCB-style)
      - quality penalty (γ): nodes that degrade success_once are penalized
    """
    nodes = _latest(load_nodes(a.campaign_dir))
    ch = _children_map(nodes)
    cands = [n for n in nodes
             if n.get("status") in (ST_OK, ST_ROOT) and n.get("objective") is not None]
    if a.max_children is not None:
        cands = [n for n in cands if len(ch[n["id"]]) < a.max_children]

    if a.composite:
        # baseline values from node #0 for normalisation
        baseline = next((n for n in nodes if n["id"] == 0), None)
        baseline_obj = baseline["objective"] if baseline else None
        baseline_headroom = baseline.get("headroom_gib") if baseline else None
        baseline_success = baseline.get("success_once") if baseline else None
        total_rounds = max((n.get("round") or 0) for n in nodes) if nodes else 0

        def composite_score(n):
            obj = n["objective"]
            score = float(obj)

            # A4.1 — potential gain from memory headroom
            # A node with headroom can still grow (raise mbs, disable offload);
            # discount its score so the frontier favours expandable nodes over
            # slightly faster but memory-tight dead ends.
            headroom = n.get("headroom_gib")
            if (headroom is not None and baseline_headroom is not None
                    and baseline_headroom > 0):
                excess_hr = max(0.0, headroom - 5.0)  # 5 GiB = min viable headroom
                potential_gain = a.alpha * excess_hr / baseline_headroom
                score = score * (1.0 - potential_gain)

            # A4.2 — exploration bonus (UCB-inspired)
            # Fewer children → less explored → higher bonus.
            nc = len(ch.get(n["id"], []))
            if total_rounds > 0:
                exploration = a.beta * math.sqrt(
                    math.log(total_rounds + 1) / (nc + 1))
                score = score - exploration

            # A4.3 — quality guard (success_once penalty)
            # A config that's faster but degrades model quality is a false win.
            n_success = n.get("success_once")
            if (n_success is not None and baseline_success is not None
                    and baseline_success > 0):
                quality_penalty = (a.gamma *
                    max(0.0, baseline_success - n_success) / baseline_success)
                score = score + quality_penalty

            return round(score, 4)

        cands.sort(key=composite_score)
    else:
        cands.sort(key=lambda n: n["objective"])

    picked = cands[: a.k]
    out = []
    for n in picked:
        rec = {"id": n["id"], "tag": n.get("tag"),
               "objective": n["objective"],
               "log_dir": n.get("log_dir"),
               "num_children": len(ch[n["id"]]),
               "headroom_gib": n.get("headroom_gib"),
               "success_once": n.get("success_once")}
        if a.composite:
            rec["composite_score"] = composite_score(n)
        out.append(rec)
    print(json.dumps({"frontier": out}))


def cmd_tree(a):
    nodes = _latest(load_nodes(a.campaign_dir))
    if not nodes:
        raise SystemExit("empty campaign")
    ch = _children_map(nodes)
    by_id = _by_id(nodes)
    roots = [n["id"] for n in nodes if n.get("parent") is None]

    def render(nid, prefix, is_last):
        n = by_id[nid]
        obj = f"{n['objective']:.4f}" if n.get("objective") is not None else "  n/a "
        conn = "└─" if is_last else "├─"
        delta = json.dumps(n.get("overrides") or {}, separators=(",", ":"))
        if delta == "{}":
            delta = "(baseline)"
        st = n.get("status")
        fm = f" {n.get('failure_mode')}" if n.get("failure_mode") else ""
        lines = [f"{prefix}{conn} #{nid} [{st}{fm}] obj={obj} {delta}"]
        kids = ch[nid]
        for i, k in enumerate(kids):
            ext = "   " if is_last else "│  "
            lines += render(k, prefix + ext, i == len(kids) - 1)
        return lines

    out = []
    for i, r in enumerate(roots):
        out += render(r, "", i == len(roots) - 1)
    print("\n".join(out))
    ok = sorted([n for n in nodes if n.get("status") == ST_OK
                 and n.get("objective") is not None], key=lambda x: x["objective"])
    if ok:
        print("\nLeaderboard (best per-trajectory time first):")
        for rank, n in enumerate(ok, 1):
            print(f"  {rank}. #{n['id']} {n.get('tag') or ''}  {n['objective']:.4f}s")


def cmd_best(a):
    nodes = _latest(load_nodes(a.campaign_dir))
    ok = [n for n in nodes if n.get("status") == ST_OK and n.get("objective") is not None]
    if not ok:
        raise SystemExit("no scored OK nodes yet")
    best = min(ok, key=lambda n: n["objective"])
    print(json.dumps({"id": best["id"], "tag": best.get("tag"),
                      "objective": best["objective"], "log_dir": best.get("log_dir"),
                      "overrides": best.get("overrides"),
                      "resolved_config": best.get("resolved_config")}, indent=2))


def cmd_exhaustion_check(a):
    """B4 combined exhaustion check — should the search stop early?

    Returns a JSON verdict: should_stop + detailed reasons.
    Stop only when ALL four conditions are met:
      1. Plateau: no improvement in best objective for 2 consecutive rounds
      2. Headroom exhausted: every frontier node has < 5 GiB headroom
      3. Offload trade exhausted: no frontier node can safely disable env offload
      4. Minimum rounds: at least 5 rounds completed
    """
    nodes = _latest(load_nodes(a.campaign_dir))
    if not nodes:
        print(json.dumps({"should_stop": False, "reason": "No nodes in campaign"}))
        return

    ch = _children_map(nodes)

    # --- candidate nodes (OK + ROOT, scored) ---
    ok_nodes = [n for n in nodes
                if n.get("status") in (ST_OK, ST_ROOT) and n.get("objective") is not None]
    if not ok_nodes:
        print(json.dumps({"should_stop": False, "reason": "No scored OK nodes yet"}))
        return

    best_obj = min(n["objective"] for n in ok_nodes)
    best_node = min(ok_nodes, key=lambda n: n["objective"])

    # --- frontier (current beam) ---
    cands = [n for n in ok_nodes
             if len(ch.get(n["id"], [])) < a.max_children]
    cands.sort(key=lambda n: n["objective"])
    frontier = cands[: a.k]

    # --- condition 1: plateau detection ---
    # Group best objective by round; check if the last 2 rounds show no improvement.
    rounds_with_best = {}
    for n in ok_nodes:
        r = n.get("round")
        if r is not None:
            rounds_with_best[r] = min(rounds_with_best.get(r, float("inf")),
                                      n["objective"])

    sorted_rounds = sorted(rounds_with_best.keys())
    plateau_rounds = 0
    if len(sorted_rounds) >= 3:
        # look at the last 2 transitions
        recent = sorted_rounds[-3:]  # last 3 rounds
        best_seq = [rounds_with_best[r] for r in recent]
        for i in range(1, len(best_seq)):
            if best_seq[i - 1] - best_seq[i] < 0.001:  # < 1 ms improvement = flat
                plateau_rounds += 1
    plateau_ok = plateau_rounds >= 2

    # --- condition 2: headroom exhausted ---
    headroom_exhausted = all(
        (n.get("headroom_gib") is not None and n.get("headroom_gib") < a.min_headroom_gib)
        for n in frontier
    )

    # --- condition 3: offload trade exhausted ---
    # The only safe offload→throughput trade is env.train.enable_offload=false.
    # rollout offload is almost always OOM under colocation; actor offload is
    # low-value.  So: if any frontier node has env offload ON and enough
    # headroom to absorb the env memory (~7 GiB typical), a trade is still open.
    offload_exhausted = True
    for n in frontier:
        cfg = n.get("resolved_config") or {}
        env_offload = cfg.get("env.train.enable_offload", True)
        if isinstance(env_offload, str):
            env_offload = env_offload.lower() in ("true", "1")
        hr = n.get("headroom_gib")
        if env_offload and hr is not None and hr >= a.min_headroom_gib:
            offload_exhausted = False
            break

    # --- condition 4: minimum rounds ---
    completed_rounds = len(sorted_rounds)
    min_rounds_ok = completed_rounds >= a.min_rounds

    should_stop = plateau_ok and headroom_exhausted and offload_exhausted and min_rounds_ok

    # build human-readable reasons
    reasons = []
    if not plateau_ok:
        reasons.append(
            f"plateau: {plateau_rounds}/2 consecutive flat rounds (need 2)")
    if not headroom_exhausted:
        fb = [(n["id"], n.get("headroom_gib")) for n in frontier
              if n.get("headroom_gib", 0) >= a.min_headroom_gib]
        reasons.append(
            f"headroom: frontier nodes {fb} still have ≥{a.min_headroom_gib} GiB")
    if not offload_exhausted:
        reasons.append("offload: env offload can still be disabled on frontier nodes")
    if not min_rounds_ok:
        reasons.append(
            f"rounds: {completed_rounds}/{a.min_rounds} minimum rounds completed")

    print(json.dumps({
        "should_stop": should_stop,
        "reasons": reasons if not should_stop else [],
        "details": {
            "plateau_rounds": plateau_rounds,
            "plateau_ok": plateau_ok,
            "headroom_exhausted": headroom_exhausted,
            "offload_exhausted": offload_exhausted,
            "min_rounds_ok": min_rounds_ok,
            "completed_rounds": completed_rounds,
            "min_rounds_required": a.min_rounds,
            "best_objective": best_obj,
            "best_node_id": best_node["id"],
            "frontier_ids": [n["id"] for n in frontier],
            "frontier_headrooms": {str(n["id"]): n.get("headroom_gib") for n in frontier},
        }
    }, indent=2))


def main():
    ap = argparse.ArgumentParser(description="Node store + tree for embodiment config search.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_campaign(p):
        p.add_argument("--campaign-dir", required=True)

    p = sub.add_parser("init"); add_campaign(p)
    p.add_argument("--log-dir", required=True)
    p.add_argument("--config", default=None, help="Path to config YAML for baseline knobs")
    p.add_argument("--overrides", default="{}")
    p.add_argument("--tag", default="baseline"); p.set_defaults(fn=cmd_init)

    p = sub.add_parser("add"); add_campaign(p)
    p.add_argument("--parent", type=int, required=True)
    p.add_argument("--overrides", required=True)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--round", type=int, default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--failure", default=None); p.set_defaults(fn=cmd_add)

    p = sub.add_parser("set-result"); add_campaign(p)
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--from-diagnosis", default=None)
    p.add_argument("--objective", type=float, default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--failure", default=None); p.set_defaults(fn=cmd_set_result)

    p = sub.add_parser("dedup"); add_campaign(p)
    p.add_argument("--parent", type=int, required=True)
    p.add_argument("--overrides", required=True); p.set_defaults(fn=cmd_dedup)

    p = sub.add_parser("frontier"); add_campaign(p)
    p.add_argument("--k", type=int, default=2)
    p.add_argument("--max-children", type=int, default=None,
                   help="retire nodes that already have >= this many children")
    p.add_argument("--composite", action=argparse.BooleanOptionalAction, default=True,
                   help="use composite scoring (A4: headroom + exploration + quality)")
    p.add_argument("--alpha", type=float, default=0.15,
                   help="headroom potential-gain coefficient (default: 0.15)")
    p.add_argument("--beta", type=float, default=0.05,
                   help="exploration bonus coefficient (default: 0.05)")
    p.add_argument("--gamma", type=float, default=0.5,
                   help="quality penalty coefficient (default: 0.5)")
    p.set_defaults(fn=cmd_frontier)

    p = sub.add_parser("tree"); add_campaign(p); p.set_defaults(fn=cmd_tree)
    p = sub.add_parser("best"); add_campaign(p); p.set_defaults(fn=cmd_best)

    p = sub.add_parser("exhaustion-check"); add_campaign(p)
    p.add_argument("--k", type=int, default=2, help="beam width (default: 2)")
    p.add_argument("--max-children", type=int, default=2,
                   help="max children per node (default: 2)")
    p.add_argument("--min-headroom-gib", type=float, default=5.0,
                   help="min headroom (GiB) to consider a node expandable (default: 5.0)")
    p.add_argument("--min-rounds", type=int, default=5,
                   help="minimum rounds before stopping (default: 5)")
    p.set_defaults(fn=cmd_exhaustion_check)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
