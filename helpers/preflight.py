#!/usr/bin/env python3
"""Preflight validation for RLinf embodiment config proposals.

Validates a *resolved* set of tunable knobs against the hard divisibility /
placement invariants enforced by rlinf/config.py, WITHOUT starting Ray or any
GPU work. Run this before launching a trial so an invalid proposal costs zero
GPU time (record it as FAILED/CONFIG_INVALID instead of burning ~15 min).

Stdlib-only. Used both as a library (`validate`, `resolve`) and a CLI:

    python preflight.py --overrides '{"actor.micro_batch_size": 40}'
    python preflight.py --resolved '{...full knob dict...}'

Exit code 0 = valid, 1 = invalid (violations printed).

The knob domains + baseline defaults mirror examples/embodiment/config/
maniskill_ppo_openvla.yaml (see reference/knob-schema.md). global_batch_size is
fixed at 640 and group_size at 1 for this recipe; both are read from the
resolved dict if present so a future un-pin still validates.
"""
from __future__ import annotations

import argparse
import json

# --- baseline (default) knob values for maniskill_ppo_openvla -----------------
BASELINE_KNOBS = {
    "cluster.component_placement.actor": "0-7",
    "cluster.component_placement.env": "0-3",
    "cluster.component_placement.rollout": "4-7",
    "env.train.total_num_envs": 128,
    "env.train.rollout_epoch": 1,
    "rollout.pipeline_stage_num": 2,
    "actor.micro_batch_size": 80,
    "actor.global_batch_size": 640,      # fixed for this recipe
    "algorithm.group_size": 1,           # fixed for this recipe
    "rollout.enable_offload": True,
    "env.train.enable_offload": True,
    "actor.enable_offload": True,
    "actor.fsdp_config.gradient_checkpointing": True,
}

# Knobs the search may propose (everything else is pinned / derived).
TUNABLE_KNOBS = {
    "cluster.component_placement.actor",
    "cluster.component_placement.env",
    "cluster.component_placement.rollout",
    "env.train.total_num_envs",
    "env.train.rollout_epoch",
    "rollout.pipeline_stage_num",
    "actor.micro_batch_size",
    "rollout.enable_offload",
    "env.train.enable_offload",
    "actor.enable_offload",
    "actor.fsdp_config.gradient_checkpointing",
}

_INT_KNOBS = {
    "env.train.total_num_envs": (1, 4096),
    "env.train.rollout_epoch": (1, 16),
    "rollout.pipeline_stage_num": (1, 16),
    "actor.micro_batch_size": (1, 4096),
}
_BOOL_KNOBS = {
    "rollout.enable_offload", "env.train.enable_offload", "actor.enable_offload",
    "actor.fsdp_config.gradient_checkpointing",
}
_PLACEMENT_KNOBS = {
    "cluster.component_placement.actor",
    "cluster.component_placement.env",
    "cluster.component_placement.rollout",
}
NUM_GPUS = 8  # single-node 8xA800 envelope


class PreflightError(ValueError):
    pass


def resolve(overrides: dict, baseline: dict | None = None) -> dict:
    """Apply an override delta on top of the baseline knobs → resolved knobs."""
    base = dict(baseline or BASELINE_KNOBS)
    for k, v in (overrides or {}).items():
        if k not in TUNABLE_KNOBS:
            raise PreflightError(f"knob not tunable / unknown: {k}")
        base[k] = v
    return base


def _parse_range(s):
    """'a-b' (inclusive) → world size (# ranks). Returns (size, lo, hi)."""
    s = str(s).strip()
    if "-" not in s:
        # single GPU index
        i = int(s)
        return 1, i, i
    lo, hi = s.split("-", 1)
    lo, hi = int(lo), int(hi)
    return hi - lo + 1, lo, hi


def _typecheck(knobs, violations):
    for k, (lo, hi) in _INT_KNOBS.items():
        v = knobs.get(k)
        if v is None:
            continue
        if isinstance(v, bool) or not isinstance(v, int):
            violations.append(f"{k}={v!r} must be an int")
        elif not (lo <= v <= hi):
            violations.append(f"{k}={v} out of range [{lo},{hi}]")
    for k in _BOOL_KNOBS:
        v = knobs.get(k)
        if v is not None and not isinstance(v, bool):
            violations.append(f"{k}={v!r} must be a bool")
    for k in _PLACEMENT_KNOBS:
        v = knobs.get(k)
        if v is None:
            continue
        try:
            size, lo, hi = _parse_range(v)
        except (ValueError, AttributeError):
            violations.append(f"{k}={v!r} is not a valid 'a-b' GPU range")
            continue
        if lo < 0 or hi > NUM_GPUS - 1 or lo > hi:
            violations.append(f"{k}={v!r} outside 0-{NUM_GPUS-1} or reversed")


def validate(resolved_knobs: dict):
    """Return (ok: bool, violations: list[str]) for a resolved knob dict."""
    v: list[str] = []
    _typecheck(resolved_knobs, v)
    if v:
        return False, v  # don't run divisibility on ill-typed values

    # world sizes from placement
    actor_world = _parse_range(resolved_knobs["cluster.component_placement.actor"])[0]
    env_world = _parse_range(resolved_knobs["cluster.component_placement.env"])[0]
    _rollout_world = _parse_range(resolved_knobs["cluster.component_placement.rollout"])[0]

    mbs = resolved_knobs["actor.micro_batch_size"]
    gbs = resolved_knobs.get("actor.global_batch_size", 640)
    tne = resolved_knobs["env.train.total_num_envs"]
    stage = resolved_knobs["rollout.pipeline_stage_num"]
    group = resolved_knobs.get("algorithm.group_size", 1)

    # actor batch: global_batch_size % (micro_batch_size * actor_world) == 0
    denom = mbs * actor_world
    if denom == 0 or gbs % denom != 0:
        v.append(f"global_batch_size({gbs}) % (micro_batch_size({mbs})*actor_world({actor_world})={denom}) != 0")

    # env divisibility
    if tne % env_world != 0:
        v.append(f"total_num_envs({tne}) % env_world({env_world}) != 0")
    else:
        per_env = tne // env_world
        if stage == 0 or per_env % stage != 0:
            v.append(f"(total_num_envs/env_world={per_env}) % pipeline_stage_num({stage}) != 0")
        else:
            chunk = per_env // stage
            if chunk < 1:
                v.append(f"total_num_envs/env_world/stage = {chunk} < 1")
            elif group and chunk % group != 0:
                v.append(f"(total_num_envs/env_world/stage={chunk}) % group_size({group}) != 0")

    return (len(v) == 0), v


def _main():
    ap = argparse.ArgumentParser(description="Preflight-validate an RLinf config proposal.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--overrides", help="JSON delta applied on baseline, e.g. '{\"actor.micro_batch_size\":40}'")
    g.add_argument("--resolved", help="JSON of a full resolved knob dict")
    ap.add_argument("--baseline", help="JSON baseline knob dict (default: built-in maniskill baseline)")
    args = ap.parse_args()

    baseline = json.loads(args.baseline) if args.baseline else None
    if args.resolved:
        knobs = json.loads(args.resolved)
    else:
        try:
            knobs = resolve(json.loads(args.overrides), baseline)
        except PreflightError as e:
            print(json.dumps({"ok": False, "violations": [str(e)], "resolved": None}, indent=2))
            raise SystemExit(1)

    ok, violations = validate(knobs)
    print(json.dumps({"ok": ok, "violations": violations,
                      "resolved": knobs}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _main()
