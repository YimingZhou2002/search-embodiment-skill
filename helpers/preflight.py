#!/usr/bin/env python3
"""Preflight validation for RLinf embodiment config proposals.

Validates a *resolved* set of tunable knobs against the hard divisibility /
placement invariants enforced by rlinf/config.py, WITHOUT starting Ray or any
GPU work. Run this before launching a trial so an invalid proposal costs zero
GPU time (record it as FAILED/CONFIG_INVALID instead of burning ~15 min).

Reads baseline knob values from a config YAML (--config), falling back to a
built-in defaults dict for any knobs not found in the file.

Stdlib+yaml. Used both as a library (load_baseline_from_config, validate, resolve)
and a CLI:

    python preflight.py --config path/to/config.yaml --overrides '{"actor.micro_batch_size": 40}'
    python preflight.py --resolved '{...full knob dict...}'

Exit code 0 = valid, 1 = invalid (violations printed).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# --- defaults for knobs not found in the config YAML ------------------------
_DEFAULT_KNOB_VALUES = {
    "cluster.component_placement.actor": "0-7",
    "cluster.component_placement.env": "0-7",
    "cluster.component_placement.rollout": "0-7",
    "env.train.total_num_envs": 64,
    "env.train.rollout_epoch": 1,
    "rollout.pipeline_stage_num": 1,
    "actor.micro_batch_size": 32,
    "actor.global_batch_size": 16384,
    "algorithm.group_size": 8,
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
    "env.train.rollout_epoch": (1, 128),
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

# --- YAML → dotted-key extraction -------------------------------------------

def _flatten_yaml(d, prefix=""):
    """Recursively flatten a nested dict to dotted-key paths.

    Handles Hydra-style comma-separated keys (e.g. ``{"actor,env,rollout": val}``)
    by splitting into multiple entries.
    """
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        # split comma-separated Hydra keys: "actor,env,rollout"
        sub_keys = [sk.strip() for sk in str(k).split(",")]
        for sk in sub_keys:
            full = f"{prefix}.{sk}" if prefix else sk
            if isinstance(v, dict) and not _is_placement_range(v):
                out.update(_flatten_yaml(v, full))
            else:
                out[full] = v
    return out


def _is_placement_range(v):
    """Detect if a dict value looks like a placement sub-map (has 'actor'/'env'/'rollout' keys
    that map to GPU ranges) — these should NOT be flattened."""
    if not isinstance(v, dict):
        return False
    placement_keys = {"actor", "env", "rollout"}
    return any(k in placement_keys for k in v.keys())


# Dotted-key → expected dotted-key in flattened YAML
_KNOB_YAML_PATHS = {
    "cluster.component_placement.actor":    "cluster.component_placement.actor",
    "cluster.component_placement.env":      "cluster.component_placement.env",
    "cluster.component_placement.rollout":  "cluster.component_placement.rollout",
    "env.train.total_num_envs":             "env.train.total_num_envs",
    "env.train.rollout_epoch":              "env.train.rollout_epoch",
    "rollout.pipeline_stage_num":           "rollout.pipeline_stage_num",
    "actor.micro_batch_size":              "actor.micro_batch_size",
    "actor.global_batch_size":             "actor.global_batch_size",
    "algorithm.group_size":                "algorithm.group_size",
    "rollout.enable_offload":              "rollout.enable_offload",
    "env.train.enable_offload":            "env.train.enable_offload",
    "actor.enable_offload":                "actor.enable_offload",
    "actor.fsdp_config.gradient_checkpointing": "actor.fsdp_config.gradient_checkpointing",
}


def _expand_placement(value):
    """Expand 'all' → '0-{NUM_GPUS-1}', pass through explicit ranges."""
    if value == "all":
        return f"0-{NUM_GPUS - 1}"
    return value


def load_baseline_from_config(config_path: str) -> dict:
    """Read a Hydra-style YAML config and extract tunable knob values.

    Returns a dict of ``{dotted_key: value}`` suitable as a baseline for
    ``resolve()``.  Any knob not found in the YAML falls back to
    ``_DEFAULT_KNOB_VALUES``.
    """
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to read config files. Install: pip install pyyaml"
        )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config file not found: {config_path}")

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    flat = _flatten_yaml(raw)

    knobs = {}
    for knob_key, yaml_path in _KNOB_YAML_PATHS.items():
        if yaml_path in flat:
            val = flat[yaml_path]
            # expand "all" placement
            if knob_key in _PLACEMENT_KNOBS:
                val = _expand_placement(val)
            knobs[knob_key] = val
        elif knob_key in _PLACEMENT_KNOBS:
            # placement may be nested under a non-flattened parent dict
            # e.g. "cluster.component_placement" = {"actor":"0-7","env":"0-3","rollout":"4-7"}
            parts = yaml_path.rsplit(".", 1)
            if len(parts) == 2 and parts[0] in flat:
                parent = flat[parts[0]]
                if isinstance(parent, dict) and parts[1] in parent:
                    knobs[knob_key] = _expand_placement(parent[parts[1]])

    # fill missing with defaults
    for k, v in _DEFAULT_KNOB_VALUES.items():
        if k not in knobs:
            knobs[k] = v

    return knobs


# --- validation --------------------------------------------------------------

class PreflightError(ValueError):
    pass


def resolve(overrides: dict, baseline: dict | None = None) -> dict:
    """Apply an override delta on top of the baseline knobs → resolved knobs."""
    base = dict(baseline or _DEFAULT_KNOB_VALUES)
    for k, v in (overrides or {}).items():
        if k not in TUNABLE_KNOBS:
            raise PreflightError(f"knob not tunable / unknown: {k}")
        base[k] = v
    return base


def _parse_range(s):
    """'a-b' (inclusive) → world size (# ranks). Returns (size, lo, hi)."""
    s = str(s).strip()
    if "-" not in s:
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


def validate(resolved_knobs: dict, max_tne_epoch_product: int | None = None):
    """Return (ok: bool, violations: list[str]) for a resolved knob dict."""
    v: list[str] = []
    _typecheck(resolved_knobs, v)
    if v:
        return False, v

    # world sizes from placement
    actor_world = _parse_range(resolved_knobs["cluster.component_placement.actor"])[0]
    env_world = _parse_range(resolved_knobs["cluster.component_placement.env"])[0]
    rollout_world = _parse_range(resolved_knobs["cluster.component_placement.rollout"])[0]

    mbs = resolved_knobs["actor.micro_batch_size"]
    gbs = resolved_knobs.get("actor.global_batch_size", 16384)
    tne = resolved_knobs["env.train.total_num_envs"]
    stage = resolved_knobs["rollout.pipeline_stage_num"]
    group = resolved_knobs.get("algorithm.group_size", 8)

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

    # rollout / actor world divisibility (invariants 7-8 from knob-schema.md)
    if rollout_world > 0 and chunk > 0 and chunk % rollout_world != 0:
        v.append(f"(total_num_envs/env_world/stage={chunk}) % rollout_world({rollout_world}) != 0")
    if actor_world > 0 and chunk > 0 and chunk % actor_world != 0:
        v.append(f"(total_num_envs/env_world/stage={chunk}) % actor_world({actor_world}) != 0")

    # tne * rollout_epoch ≤ 8× baseline
    if max_tne_epoch_product is not None:
        tne = resolved_knobs["env.train.total_num_envs"]
        epoch = resolved_knobs["env.train.rollout_epoch"]
        product = tne * epoch
        if product > max_tne_epoch_product:
            v.append(
                f"total_num_envs({tne}) * rollout_epoch({epoch}) = {product} "
                f"> max allowed ({max_tne_epoch_product} = 8× baseline)"
            )

    return (len(v) == 0), v


# --- CLI --------------------------------------------------------------------

def _main():
    ap = argparse.ArgumentParser(description="Preflight-validate an RLinf config proposal.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--overrides", help="JSON delta applied on baseline, e.g. '{\"actor.micro_batch_size\":40}'")
    g.add_argument("--resolved", help="JSON of a full resolved knob dict")
    ap.add_argument("--config", help="Path to config YAML to read baseline knobs from (e.g. examples/embodiment/config/libero_10_grpo_openvlaoft.yaml)")
    ap.add_argument("--baseline", help="JSON baseline knob dict (overrides --config)")
    args = ap.parse_args()

    # resolve baseline
    if args.baseline:
        baseline = json.loads(args.baseline)
    elif args.config:
        baseline = load_baseline_from_config(args.config)
    else:
        baseline = None  # use _DEFAULT_KNOB_VALUES

    # compute max tne*epoch product = 8 × baseline
    baseline_for_product = baseline if baseline is not None else _DEFAULT_KNOB_VALUES
    base_tne = baseline_for_product.get("env.train.total_num_envs", 64)
    base_epoch = baseline_for_product.get("env.train.rollout_epoch", 1)
    max_tne_epoch_product = base_tne * base_epoch * 8

    if args.resolved:
        knobs = json.loads(args.resolved)
    else:
        try:
            knobs = resolve(json.loads(args.overrides), baseline)
        except PreflightError as e:
            print(json.dumps({"ok": False, "violations": [str(e)], "resolved": None}, indent=2))
            raise SystemExit(1)

    ok, violations = validate(knobs, max_tne_epoch_product=max_tne_epoch_product)
    print(json.dumps({"ok": ok, "violations": violations,
                      "resolved": knobs}, indent=2))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    _main()
