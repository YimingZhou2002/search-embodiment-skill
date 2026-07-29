#!/usr/bin/env python3
"""Node store + tree for the embodiment config search (beam search).

Maintains, per campaign directory:
  * nodes.jsonl  — append-only log of nodes (source of truth, resumable)
  * tree.json    — materialized view: each node with explicit parent + children[]

Each node maps: id -> config (overrides delta + resolved knobs) -> log_dir, plus
parent/children tree links, objective (step_time_per_traj_s; lower is better),
and status. Deterministic bookkeeping only — proposals are made by the caller
(the skill's LLM step). Stdlib-only.

CLI:
    search_store.py init      --campaign-dir D --log-dir L [--overrides '{}'] [--tag baseline]
    search_store.py add       --campaign-dir D --parent P --overrides '<json>' --log-dir L [--round R] [--tag T]
    search_store.py set-result --campaign-dir D --id I (--from-diagnosis F | --objective X [--status S] [--failure M])
    search_store.py dedup     --campaign-dir D --parent P --overrides '<json>'
    search_store.py frontier  --campaign-dir D [--k 2]
    search_store.py tree      --campaign-dir D
    search_store.py best      --campaign-dir D

Config resolution + hashing reuse preflight.resolve so the SHA matches what
preflight validates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import preflight  # sibling helper: resolve(), BASELINE_KNOBS

NODES_FILE = "nodes.jsonl"
TREE_FILE = "tree.json"

# statuses / failure modes
ST_ROOT, ST_OK, ST_FAILED, ST_DUPLICATE = "ROOT", "OK", "FAILED", "DUPLICATE"


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


def _resolve_for(nodes_by_id, parent_id, overrides):
    """Cumulative resolved knobs = baseline + chain of ancestor overrides + this delta."""
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
    resolved = dict(preflight.BASELINE_KNOBS)
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
        })
    ok = [n for n in nodes if n.get("status") == ST_OK and n.get("objective") is not None]
    for n in sorted(ok, key=lambda x: x["objective"]):
        view["leaderboard"].append({"id": n["id"], "tag": n.get("tag"),
                                    "objective": n["objective"]})
    with open(os.path.join(campaign_dir, TREE_FILE), "w") as f:
        json.dump(view, f, indent=2)


# --- commands ----------------------------------------------------------------
def cmd_init(a):
    os.makedirs(a.campaign_dir, exist_ok=True)
    if load_nodes(a.campaign_dir):
        raise SystemExit(f"campaign already initialized: {_nodes_path(a.campaign_dir)}")
    overrides = json.loads(a.overrides) if a.overrides else {}
    resolved = _resolve_for({}, None, overrides)
    node = {"id": 0, "parent": None, "overrides": overrides,
            "resolved_config": resolved, "config_sha": _sha(resolved),
            "log_dir": os.path.abspath(a.log_dir), "objective": None,
            "status": ST_ROOT, "failure_mode": None, "round": 0,
            "tag": a.tag or "baseline"}
    _append_node(a.campaign_dir, node)
    _write_tree(a.campaign_dir, load_nodes(a.campaign_dir))
    print(json.dumps({"id": 0, "config_sha": node["config_sha"]}))


def cmd_add(a):
    nodes = load_nodes(a.campaign_dir)
    if not nodes:
        raise SystemExit("campaign not initialized; run `init` first")
    by_id = _by_id(nodes)
    if a.parent not in by_id:
        raise SystemExit(f"unknown parent id: {a.parent}")
    overrides = json.loads(a.overrides)
    resolved = _resolve_for(by_id, a.parent, overrides)
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
    if a.from_diagnosis:
        d = json.load(open(a.from_diagnosis))
        eff = d.get("efficiency", {})
        objective = eff.get("step_time_per_traj_s")
        if eff.get("likely_oom_before_first_step"):
            status, failure = ST_FAILED, "OOM"
        elif objective is None:
            status, failure = ST_FAILED, "METRICS_MISSING"
        else:
            status = ST_OK
    # rewrite: append an updated copy (append-only log; latest wins on load)
    node = dict(by_id[a.id])
    if objective is not None:
        node["objective"] = objective
    if status:
        node["status"] = status
    node["failure_mode"] = failure
    _append_node(a.campaign_dir, node)
    # collapse duplicates by id (latest wins) for the tree view
    _write_tree(a.campaign_dir, _latest(load_nodes(a.campaign_dir)))
    print(json.dumps({"id": a.id, "objective": node["objective"],
                      "status": node["status"], "failure_mode": node["failure_mode"]}))


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
    resolved = _resolve_for(by_id, a.parent, overrides)
    sha = _sha(resolved)
    for n in nodes:
        if n.get("config_sha") == sha:
            print(json.dumps({"duplicate": True, "of_id": n["id"],
                              "config_sha": sha, "objective": n.get("objective"),
                              "status": n.get("status")}))
            return
    print(json.dumps({"duplicate": False, "config_sha": sha}))


def cmd_frontier(a):
    """Best-K OK nodes by objective (the beam). Not leaf-restricted — a strong
    node stays expandable even if a prior child failed. `--max-children` retires
    a node once it already has that many children (so the beam moves on to the
    next-best instead of re-expanding an exhausted node)."""
    nodes = _latest(load_nodes(a.campaign_dir))
    ch = _children_map(nodes)
    cands = [n for n in nodes
             if n.get("status") in (ST_OK, ST_ROOT) and n.get("objective") is not None]
    if a.max_children is not None:
        cands = [n for n in cands if len(ch[n["id"]]) < a.max_children]
    cands.sort(key=lambda n: n["objective"])
    picked = cands[: a.k]
    print(json.dumps({"frontier": [{"id": n["id"], "tag": n.get("tag"),
                                    "objective": n["objective"],
                                    "log_dir": n.get("log_dir"),
                                    "num_children": len(ch[n["id"]])} for n in picked]}))


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
    # leaderboard
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


def main():
    ap = argparse.ArgumentParser(description="Node store + tree for embodiment config search.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_campaign(p):
        p.add_argument("--campaign-dir", required=True)

    p = sub.add_parser("init"); add_campaign(p)
    p.add_argument("--log-dir", required=True); p.add_argument("--overrides", default="{}")
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
    p.set_defaults(fn=cmd_frontier)

    p = sub.add_parser("tree"); add_campaign(p); p.set_defaults(fn=cmd_tree)
    p = sub.add_parser("best"); add_campaign(p); p.set_defaults(fn=cmd_best)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
