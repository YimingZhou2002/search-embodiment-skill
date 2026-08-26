#!/usr/bin/env python3
"""Wiki index & query tool for the embodiment config search (self-learning).

Maintains INDEX.md files for the wiki directory tree and provides
query/verify utilities. Wiki entries themselves are written by the
search agent using the template at reference/wiki-entry-template.md.

CLI:
    wiki_index.py query   --wiki-dir W [--env E] [--model M] [--algorithm A]
    wiki_index.py verify  --campaign-dir D --wiki-dir W
    wiki_index.py index   --wiki-dir W
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import date
from typing import Any


# ---------------------------------------------------------------------------
# Data loading (used by verify)
# ---------------------------------------------------------------------------

def load_nodes(campaign_dir: str) -> list[dict]:
    """Load nodes from nodes.jsonl."""
    path = os.path.join(campaign_dir, "nodes.jsonl")
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


def _latest(nodes: list[dict]) -> list[dict]:
    """Collapse append-only log so the last record per id wins."""
    out = {}
    for n in nodes:
        out[n["id"]] = n
    return [out[k] for k in sorted(out)]


def load_tree(campaign_dir: str) -> dict:
    """Load tree.json."""
    path = os.path.join(campaign_dir, "tree.json")
    if not os.path.exists(path):
        return {"nodes": [], "leaderboard": []}
    return json.load(open(path))


def load_diagnosis_json(node_dir: str) -> dict | None:
    """Load diagnosis.json from a node directory."""
    path = os.path.join(node_dir, "diagnosis.json")
    if not os.path.exists(path):
        return None
    try:
        return json.load(open(path))
    except (json.JSONDecodeError, OSError):
        return None


def extract_memory(diagnosis: dict | None) -> dict:
    """Extract memory data from diagnosis.json."""
    if not diagnosis:
        return {"max_used_gib": 0, "total_gib": 80, "max_used_pct": 0, "oom_risk": "unknown"}
    mem = diagnosis.get("memory", {})
    return {
        "max_used_gib": mem.get("max_used_gib", 0),
        "total_gib": mem.get("total_gib", 80),
        "max_used_pct": mem.get("max_used_pct", 0),
        "oom_risk": mem.get("oom_risk", "unknown"),
    }


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------

def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _write_relative(path: str, content: str):
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        f.write(content + "\n")


# ---------------------------------------------------------------------------
# INDEX generation
# ---------------------------------------------------------------------------

def _dimension_has_entries(wiki_dir: str, dim: str) -> bool:
    """Check if a dimension directory has any entries."""
    dim_path = os.path.join(wiki_dir, dim)
    if not os.path.exists(dim_path):
        return False
    for name in os.listdir(dim_path):
        child = os.path.join(dim_path, name)
        if os.path.isdir(child):
            # Subdirectory entries (e.g. env/maniskill/ppo-openvlaoft.md)
            for fname in os.listdir(child):
                if fname.endswith(".md") and fname != "INDEX.md":
                    return True
        elif name.endswith(".md") and name != "INDEX.md":
            # Direct entries (e.g. knob-effect/enable_offload.md)
            return True
    return False


def _write_leaf_index(wiki_dir: str, dim: str, name: str):
    """Write the per-leaf INDEX.md."""
    leaf_path = os.path.join(wiki_dir, dim, name)
    entries = []
    if os.path.exists(leaf_path):
        for fname in sorted(os.listdir(leaf_path)):
            if fname.endswith(".md") and fname != "INDEX.md":
                entries.append(fname)

    lines = [
        f"# {dim}: {name}",
        "",
        f"Last updated: {date.today()}",
        "",
        "## Entries",
        "",
    ]
    if entries:
        for e in entries:
            display = e.replace(".md", "")
            lines.append(f"- [{display}]({e})")
    else:
        lines.append("No entries yet.")
    lines.extend([
        "",
        f"[Back to {dim} index](../INDEX.md)",
        "",
    ])
    _write_relative(os.path.join(leaf_path, "INDEX.md"), "\n".join(lines))


def _write_dimension_index(wiki_dir: str, dim: str):
    """Write the per-dimension INDEX.md."""
    dim_path = os.path.join(wiki_dir, dim)
    entries = []
    direct_files = []
    if os.path.exists(dim_path):
        for name in sorted(os.listdir(dim_path)):
            child = os.path.join(dim_path, name)
            if os.path.isdir(child):
                # Subdirectory entries (env, model, algorithm, cfg)
                has_entry = any(
                    f.endswith(".md") and f != "INDEX.md"
                    for f in os.listdir(child)
                )
                if has_entry:
                    entries.append(name)
            elif name.endswith(".md") and name != "INDEX.md":
                # Direct entries (knob-effect)
                direct_files.append(name)

    lines = [
        f"# Wiki Index: {dim}",
        "",
        f"Last updated: {date.today()}",
        "",
        "## Entries",
        "",
    ]
    if entries or direct_files:
        lines.append("| Name | Description |")
        lines.append("|------|-------------|")
        for name in entries:
            lines.append(f"| [{name}]({name}/INDEX.md) | {dim} |")
        for name in direct_files:
            display = name.replace(".md", "")
            lines.append(f"| [{display}]({name}) | {dim} |")
    else:
        lines.append("No entries yet.")
    lines.extend([
        "",
        "## Cross-Dimension Links",
        "",
    ])
    for other_dim in ["env", "model", "algorithm", "cfg", "knob-effect"]:
        if other_dim != dim:
            lines.append(f"- [{other_dim}](../{other_dim}/INDEX.md)")
    lines.append("")
    _write_relative(os.path.join(dim_path, "INDEX.md"), "\n".join(lines))


def _write_top_index(wiki_dir: str):
    """Write the top-level INDEX.md."""
    lines = [
        "# Wiki Index -- RLinf Embodiment Config Search Experience",
        "",
        f"Last updated: {date.today()}",
        "",
        "## Dimensions",
        "",
    ]
    for dim, desc in [
        ("env", "Per-environment tuning experience"),
        ("model", "Per-model tuning experience"),
        ("algorithm", "Per-algorithm tuning experience"),
        ("cfg", "Per-config-pattern tuning experience"),
        ("knob-effect", "Cross-campaign knob effect experience"),
    ]:
        if _dimension_has_entries(wiki_dir, dim):
            lines.append(f"- [{dim}]({dim}/INDEX.md) -- {desc}")
        else:
            lines.append(f"- {dim} -- {desc} (no entries yet)")

    lines.extend([
        "",
        "## Usage",
        "",
        "When proposing knob deltas, query the wiki for relevant entries:",
        "",
        "```bash",
        'python "$SKILL/wiki/wiki_index.py" query --wiki-dir "$SKILL/wiki" \\',
        "  --env <env> --model <model> --algorithm <algorithm>",
        "```",
        "",
    ])
    _write_relative(os.path.join(wiki_dir, "INDEX.md"), "\n".join(lines))


def generate_index(wiki_dir: str):
    """Regenerate all INDEX.md files."""
    # First write per-leaf indexes (so dimension indexes can reference them)
    for dim in ("env", "model", "algorithm"):
        dim_path = os.path.join(wiki_dir, dim)
        if not os.path.exists(dim_path):
            continue
        for name in os.listdir(dim_path):
            leaf_path = os.path.join(dim_path, name)
            if os.path.isdir(leaf_path):
                _write_leaf_index(wiki_dir, dim, name)

    # Then write per-dimension indexes
    for dim in ("env", "model", "algorithm", "cfg", "knob-effect"):
        dim_path = os.path.join(wiki_dir, dim)
        if os.path.exists(dim_path):
            _write_dimension_index(wiki_dir, dim)

    # Write top-level index LAST (so it can reference the dimension indexes)
    _write_top_index(wiki_dir)


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------

def query_entries(
    wiki_dir: str, env: str | None = None, model: str | None = None,
    algorithm: str | None = None,
) -> list[str]:
    """Query wiki entries by dimension, returning paths ranked by relevance."""
    matches: list[tuple[int, str]] = []

    # 1. Exact match: env/<env>/<algo>-<model>.md (best relevance)
    if env and algorithm and model:
        path = os.path.join(wiki_dir, "env", env, f"{algorithm}-{model}.md")
        if os.path.exists(path):
            matches.append((0, path))

    # 2. Env match: env/<env>/INDEX.md
    if env:
        path = os.path.join(wiki_dir, "env", env, "INDEX.md")
        if os.path.exists(path):
            matches.append((1, path))

    # 3. Model match: model/<model>/<env>-<algo>.md
    if model and env and algorithm:
        path = os.path.join(wiki_dir, "model", model, f"{env}-{algorithm}.md")
        if os.path.exists(path):
            matches.append((2, path))

    # 4. Algorithm match: algorithm/<algo>/<env>-<model>.md
    if algorithm and env and model:
        path = os.path.join(wiki_dir, "algorithm", algorithm, f"{env}-{model}.md")
        if os.path.exists(path):
            matches.append((3, path))

    # 5. All env entries for this env
    if env:
        env_dir = os.path.join(wiki_dir, "env", env)
        if os.path.exists(env_dir):
            for fname in sorted(os.listdir(env_dir)):
                if fname.endswith(".md") and fname != "INDEX.md":
                    path = os.path.join(env_dir, fname)
                    if not any(p == path for _, p in matches):
                        matches.append((4, path))

    matches.sort()
    return [m[1] for m in matches]


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_campaign(campaign_dir: str, wiki_dir: str) -> list[dict]:
    """Verify consistency of campaign data and wiki entries. Returns list of issues."""
    issues = []
    nodes_raw = load_nodes(campaign_dir)
    nodes = _latest(nodes_raw)
    tree = load_tree(campaign_dir)
    ok_nodes = [n for n in nodes if n.get("status") == "OK" and n.get("objective") is not None]
    ok_nodes.sort(key=lambda n: n["objective"])

    # 1. Check internal consistency
    tree_leaderboard = tree.get("leaderboard", [])
    if tree_leaderboard and ok_nodes:
        tree_ids = [n["id"] for n in tree_leaderboard[:3]]
        ok_ids = [n["id"] for n in ok_nodes[:3]]
        if tree_ids != ok_ids:
            issues.append({
                "severity": "warning",
                "check": "leaderboard_consistency",
                "detail": f"Tree leaderboard top-3 ({tree_ids}) differs from actual top-3 ({ok_ids})",
            })

    # 2. Check cross-node consistency
    if len(ok_nodes) >= 3:
        objs = [n.get("objective", 0) for n in ok_nodes[:5]]
        obj_range = max(objs) - min(objs)
        obj_mean = sum(objs) / len(objs)
        if obj_mean > 0 and obj_range / obj_mean > 0.5:
            issues.append({
                "severity": "warning",
                "check": "cross_node_consistency",
                "detail": f"Large variance in top-5 objectives: range={obj_range:.4f}, mean={obj_mean:.4f}",
            })

    # 3. Check memory model sanity
    if ok_nodes and ok_nodes[0].get("log_dir"):
        diag = load_diagnosis_json(ok_nodes[0]["log_dir"])
        if diag:
            mem = extract_memory(diag)
            total = mem.get("max_used_gib", 0)
            gpu_total = mem.get("total_gib", 80)
            if total > gpu_total * 1.1:
                issues.append({
                    "severity": "error",
                    "check": "memory_model_sanity",
                    "detail": (
                        f"Peak memory ({total:.0f} GiB) exceeds "
                        f"GPU capacity ({gpu_total} GiB)"
                    ),
                })

    # 4. Check OOM consistency
    for fn in [n for n in nodes if n.get("status") == "FAILED"]:
        failure = fn.get("failure_mode", "unknown")
        if fn.get("log_dir"):
            diag = load_diagnosis_json(fn["log_dir"])
            if diag:
                eff = diag.get("efficiency", {})
                if eff.get("likely_oom_before_first_step") and failure != "OOM":
                    issues.append({
                        "severity": "info",
                        "check": "failure_mode_consistency",
                        "detail": f"Node #{fn['id']} has likely_oom but failure_mode={failure}",
                    })

    return issues


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------

def cmd_query(args):
    """Query wiki entries by dimension."""
    paths = query_entries(args.wiki_dir, args.env, args.model, args.algorithm)
    for p in paths:
        rel = os.path.relpath(p, args.wiki_dir)
        print(rel)


def cmd_verify(args):
    """Verify campaign data consistency."""
    issues = verify_campaign(args.campaign_dir, args.wiki_dir)
    if not issues:
        print("No issues found.")
        return
    for issue in issues:
        sev = issue["severity"]
        check = issue["check"]
        detail = issue["detail"]
        if sev == "error":
            print(f"ERROR [{check}] {detail}")
        elif sev == "warning":
            print(f"WARN  [{check}] {detail}")
        else:
            print(f"INFO  [{check}] {detail}")


def cmd_index(args):
    """Regenerate all INDEX.md files."""
    generate_index(args.wiki_dir)
    print(f"INDEX.md files regenerated in {args.wiki_dir}")


def main():
    ap = argparse.ArgumentParser(
        description="Wiki index & query tool for embodiment config search."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_wiki(p):
        p.add_argument(
            "--wiki-dir", required=True,
            help="Path to the wiki directory (e.g. $SKILL/wiki)",
        )

    p = sub.add_parser("query")
    add_wiki(p)
    p.add_argument("--env", default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--algorithm", default=None)
    p.set_defaults(fn=cmd_query)

    p = sub.add_parser("verify")
    p.add_argument("--campaign-dir", required=True)
    add_wiki(p)
    p.set_defaults(fn=cmd_verify)

    p = sub.add_parser("index")
    add_wiki(p)
    p.set_defaults(fn=cmd_index)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()