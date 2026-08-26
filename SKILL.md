---
name: search-embodiment
description: Search for the best-performing RLinf embodiment config via a beam search over config knobs, profiling each trial. Use when the user wants to auto-tune / optimize / search embodiment training configs (e.g. "find the best config for maniskill_ppo_openvla", "tune the embodiment run", "search configs"). Cold-starts from the baseline, then each round expands the 2 best nodes with 2 proposed knob deltas each (4 profiled trials/round), maintaining a node tree and leaderboard.
---

# search-embodiment

Beam search (width 2, branching 2) over RLinf embodiment config knobs. Each trial
is a short profiled run; the objective is **`step_time_per_traj_s`** from
`diagnosis.json` (lower = faster). Maintains a node tree (`nodes.jsonl` +
`tree.json`) mapping **node id → config → log dir** and **node → parent →
children**.

> **Golden rule:** minimize `step_time_per_traj_s` under preflight + OOM constraints.
> Short trials. Never launch a duplicate or preflight-invalid config. Propose from
> evidence (each node's `diagnosis.json`), never guess. **Every OK trial MUST produce
> a `REPORT.md` before the next trial launches — no exceptions.**

## 1. Configuration

`ROUNDS=10`, `BEAM=2` (frontier size), `BRANCH=2` (proposals/node), `STEPS=2`.

4 trials/round × up to 10 rounds ≈ **40 runs, sequential** (each needs all GPUs;
`gpu_check.sh` enforces one at a time), ~15 min each ≈ **~10 hours**. Use short
trials (`STEPS` small). The campaign is **resumable** — `nodes.jsonl` is append-only,
so a killed run continues from the last recorded node. Confirm the user wants a
full campaign before starting; offer a smaller round/beam count for a quick pass.

The **objective** is per-trajectory time, comparable across trials of different
`total_num_envs` / batch (see `reference/concepts.md` §1), so short trials are valid
for ranking.

## 2. Paths

- `PROJECT` = `$(pwd)` at invocation (repo folder).
- `RLINF_ROOT` = `${RLINF_ROOT:-$PROJECT/RLinf}`.
- `SKILL` = `$PROJECT/.claude/skills/search-embodiment` .
- `VENV` = `${EMBODIMENT_VENV:-${VIRTUAL_ENV:-/root/venv/openvla}}`.
- `WIKI` = `$SKILL/wiki` — holds cross-campaign wiki entries (env, model, algorithm, cfg, knob-effect).
- `CAMPAIGN` = `$RLINF_ROOT/logs/search_<timestamp>/` — holds `nodes.jsonl`,
  `tree.json`, `SEARCH_REPORT.md`, and per-trial dirs `node_<id>-<tag>/`.
- `CONFIG` = the config to tune (default `maniskill_ppo_openvla`, or the skill arg).
- `STEPS` = trial length (default small, e.g. `2`; overridable).

## 3. Worker-level atomic modeling (cold-start profiling)

> **⏰ Mandatory step — run this once, right after the baseline, before round 1.**
> The search loop depends on these curves. Without them, every proposal is a guess.

Beyond whole-config trials, RLinf ships a **per-component atomic-modeling harness**
(`examples/embodiment/run_sweep.sh` → `benchmark_sweep.py`) that isolates the three
workers on **disjoint GPUs** and sweeps each one's scaling knob independently, so you
get a clean cost curve per component without cross-component interference. It measures
compute time and VRAM/HBM (plus env CPU% / host-RAM / GPU-util) for: **env**
`num_envs`, **rollout** batch size, and **actor** `micro_batch_size` (with
`global_batch_size` pinned). Seeds are captured inline; results land in
`<log_dir>/bench_msgs/sweep_{env_interact,rollout_predict,actor_micro_batch}.json|csv`.

Pin the output under the campaign dir so the curves live with the search, then read
each component's curve to seed proposals: does step time / peak HBM grow sub-linearly
(headroom to push the knob) or is it already saturated / near-OOM (back off)? Every
round's `micro_batch_size` / batch / env-parallelism proposal should cite the matching
curve and cross-check the target point against the sweep's OOM/peak-HBM column, rather
than guessing.

```bash
# one-off at cold start, per CONFIG; disjoint-GPU sweep of env/rollout/actor.
# GPU-exclusive like a trial — gpu_check first.
bash "$SKILL/scripts/gpu_check.sh" || { echo "GPUs busy — wait"; }
cd "$RLINF_ROOT" && source "$VENV/bin/activate" && \
RLINF_BENCH_OUT="$CAMPAIGN/atomic_model" RLINF_BENCH_ENV_MAX=64 \
  bash examples/embodiment/run_sweep.sh <CONFIG>
```

Caveat (RoboTwin/SAPIEN): the **env** axis builds a fresh `num_envs`-wide sim per point
on a single GPU and leaks TLS pthread-keys, so large `num_envs` points can hit an
**uncatchable native SIGABRT**. Bound it with `RLINF_BENCH_ENV_MAX` (RoboTwin: `64`);
already-measured rows survive in `sweep_env_interact.partial.jsonl`. See
`examples/embodiment/run_sweep.sh` header for all knobs. This harness **profiles/models
the workers only** — it does not launch or register search nodes; the beam search still
drives the actual trials.

## 4. Launch a trial (the per-node eval)

A trial = one short profiled run with a knob delta, then plot + diagnose. Run this as
a **single background** Bash command (GPU-exclusive; `gpu_check` first), then plot and
diagnose. `OV` is the space-joined hydra tokens for the delta
(e.g. `actor.micro_batch_size=40 env.train.enable_offload=false`).

```bash
PROJECT="$(pwd)"; RLINF_ROOT="${RLINF_ROOT:-$PROJECT/RLinf}"
SKILL="$PROJECT/.claude/skills/search-embodiment"
VENV="${EMBODIMENT_VENV:-${VIRTUAL_ENV:-/root/venv/openvla}}"
NODE_DIR="$CAMPAIGN/node_<id>-<tag>"

bash "$SKILL/scripts/gpu_check.sh" || { echo "GPUs busy — wait"; }   # gate

# launch (background: run_in_background=true)
cd "$RLINF_ROOT" && \
source "$VENV/bin/activate" && \
source "$SKILL/profiler/enable_nvitop.sh" && \
source "$SKILL/profiler/enable_timeline.sh" && \
STEPS=2 RUN_LOG_DIR="$NODE_DIR" EXTRA_OVERRIDES="<OV>" \
  bash examples/embodiment/run_embodiment.sh <CONFIG>

# after it exits: plot + diagnose (writes NODE_DIR/diagnosis.json)
python "$SKILL/profiler/plot_timeline.py" "$NODE_DIR/timeline" -o "$NODE_DIR/timeline.png"  --format png
python "$SKILL/profiler/plot_timeline.py" "$NODE_DIR/timeline" -o "$NODE_DIR/timeline.html"  --format html
python "$SKILL/profiler/plot_nvitop.py"   "$NODE_DIR/nvitop"   -o "$NODE_DIR/nvitop.png"    --summary-output --format png
python "$SKILL/profiler/plot_nvitop.py"   "$NODE_DIR/nvitop"   -o "$NODE_DIR/nvitop.html"    --format html
python "$SKILL/helpers/diagnose.py" "$NODE_DIR"
```

`RUN_LOG_DIR` pins the trial's log dir and `EXTRA_OVERRIDES` injects the knob delta —
both supported by `run_embodiment.sh` (additive passthrough). `diagnose.py` needs
`nvitop_summary.log`, which `plot_nvitop.py` writes — so plot before diagnosing (as
here). A trial that OOMs writes no `metrics.log`; `diagnose.py` sets
`likely_oom_before_first_step`, and `set-result` records it as FAILED/OOM.

After the shell exits and `diagnosis.json` is written, proceed to
[§5 Write per-node REPORT.md](#5-write-per-node-reportmd).

**External dependency** on the additive `RUN_LOG_DIR`/`EXTRA_OVERRIDES` passthrough
in `RLinf/examples/embodiment/run_embodiment.sh`.

## 5. Write per-node REPORT.md

This section is referenced by the round loop in §6.2. After every OK trial, write
`$NODE_DIR/REPORT.md` **immediately** — do not defer to later.

1. Read `$NODE_DIR/diagnosis.json` (the extracted signals) and skim `diagnosis.txt`.
2. Read [`reference/concepts.md`](reference/concepts.md) (domain model — per-trajectory
   metric, generation nesting, offload/OOM memory model) and
   [`reference/diagnosis-playbook.md`](reference/diagnosis-playbook.md) (signal → cause →
   fix). Match the run's signals to patterns.
3. Fill [`reference/report-template.md`](reference/report-template.md) with the actual
   numbers — cite each `diagnosis.json` field behind every claim, rank optimization
   directions by per-trajectory-time impact, and save to `$NODE_DIR/REPORT.md`.

Report the headline to the user (per-trajectory time + the dominant bottleneck pattern)
and the `REPORT.md` path. Do **not** invent numbers — everything traces to
`diagnosis.json`,`metrics.log`, nvitop and timeline profile results.

## 6. Workflow

### 0. Set up the campaign
Pick `CAMPAIGN=$RLINF_ROOT/logs/search_$(date +%Y%m%d-%H%M%S)` and `mkdir -p` it.

### 1. Cold start — evaluate the baseline
Run the baseline config as one node (see [§4 Launch a trial](#4-launch-a-trial-the-per-node-eval)) with **no
overrides**, log dir `$CAMPAIGN/node_0-baseline`. After it's diagnosed, register
the root, then **immediately write its `REPORT.md`** (see [§5 Write per-node REPORT.md](#5-write-per-node-reportmd)):

```bash
python "$SKILL/helpers/search_store.py" init \
  --campaign-dir "$CAMPAIGN" --log-dir "$CAMPAIGN/node_0-baseline" --tag baseline
python "$SKILL/helpers/search_store.py" set-result \
  --campaign-dir "$CAMPAIGN" --id 0 --from-diagnosis "$CAMPAIGN/node_0-baseline/diagnosis.json"
```

### 2. Atomic modeling — per-component cost curves

> **⚠️ DANGER ZONE — atomic modeling is the most-skipped step and proposals depend on it.**
> Every round's `micro_batch_size` / batch / env-parallelism proposal expects the atomic
> sweep's per-component curves. Without them, you are guessing, which is exactly what the
> golden rule forbids. **Do not proceed to step 3 (Rounds) without running this step.**

Run the worker-level atomic-modeling sweep **once** to build per-component cost/scaling
models for this config — see
[§3 Worker-level atomic modeling](#3-worker-level-atomic-modeling-cold-start-profiling).
The whole-config baseline tells you the *current* bottleneck; the atomic sweep tells you
*how each component scales* with its knob, which is what the proposals in every round
draw on. Keep its curves alongside the campaign.

**Verify the sweep completed before proceeding to §6.3 (Rounds):**
```bash
# A minimal check: the sweep should have produced at least one JSON/CSV result.
# If this finds nothing, re-run the sweep before continuing.
ls "$CAMPAIGN/atomic_model/bench_msgs/sweep_"*.{json,csv} 2>/dev/null \
  || echo "MISSING: atomic-modeling data not found. Run the sweep now."
```

### 3. Rounds

**Before round 1:** verify the atomic-modeling sweep was run (see §6.2). If the data
is missing, the agent will guess at proposals — **stop and run step 2 first:**
```bash
ls "$CAMPAIGN/atomic_model/bench_msgs/sweep_"*.{json,csv} 2>/dev/null \
  || { echo "STOP: atomic-modeling data missing. Run step 2 first."; exit 1; }
```

For `r` in `1..ROUNDS`:

#### Frontier
1. **Frontier** — the best `BEAM` nodes still expandable:
   ```bash
   python "$SKILL/helpers/search_store.py" frontier --campaign-dir "$CAMPAIGN" --k 2 --max-children 2
   ```

#### Propose
2. **Propose** — for each frontier node, first read the wiki for relevant past experience,
   then read `<node.log_dir>/diagnosis.json` and **`REPORT.md`**:
   a. **Read wiki entries** relevant to this config's env, model, and algorithm:
      ```bash
      python "$WIKI/wiki_index.py" query \
        --wiki-dir "$WIKI" \
        --env "$(echo $CONFIG | sed 's/_.*//')" \
        --model "$(echo $CONFIG | sed 's/.*_//')" \
        --algorithm "$(echo $CONFIG | awk -F_ '{for(i=1;i<=NF;i++)if($i~/^(ppo|grpo|sac|iql|dsrl|opd|crossq|rlt|dagger)$/)print $i}')"
      ```
      This returns paths to relevant wiki entries. Read the top 1–3 entries to:
      - Prioritize knobs known to be effective for this env/model/algo combination
      - Avoid known dead ends (OOM walls, invalid configs)
      - Get component memory footprint for offload decisions
      If the wiki has high-confidence entries that contradict a proposed knob direction,
      do not propose it. If wiki entries confirm a knob direction, prioritize it.
   b. Read `<node.log_dir>/diagnosis.json` and `REPORT.md` (both must exist; if
      REPORT.md is missing, stop and write it first — the diagnosis and report together
      are the evidence for the next proposals).
   c. Propose `BRANCH` knob deltas using
      [`reference/search-recipe.md`](reference/search-recipe.md),
      [`reference/knob-schema.md`](reference/knob-schema.md), and
      [`reference/diagnosis-playbook.md`](reference/diagnosis-playbook.md).
      Incorporate wiki knowledge: if the wiki says a knob direction is effective for this
      env/model, propose it with higher priority. If the wiki says a knob is a dead end, skip it.
   Each delta is a small JSON of hydra overrides, e.g.
   `{"env.train.enable_offload": false}`. For knobs whose effect is a per-component
   **scaling curve** (env `num_envs`, rollout batch size, actor `micro_batch_size`),
   prefer proposing from the atomic-modeling sweep's measured curves (built once at
   cold start) over guessing — see
   [§3 Worker-level atomic modeling](#3-worker-level-atomic-modeling-cold-start-profiling).

#### For each proposal
3. **For each proposal** `d` on parent `P` (up to `BEAM×BRANCH` = 4):
   a. **dedup:** `search_store.py dedup --campaign-dir "$CAMPAIGN" --parent P --overrides '<d>'`.
      If `duplicate`, skip the run (reuse the known objective); optionally record a
      `--status DUPLICATE` node for provenance. Propose a different delta.
   b. **preflight:** `python "$SKILL/helpers/preflight.py" --config "$RLINF_ROOT/examples/embodiment/config/<CONFIG>.yaml" --overrides '<d>'`
      This loads the real config's baseline knobs (e.g. `global_batch_size`, `total_num_envs`) so validation uses actual values, not hardcoded defaults. If invalid, record it without running:
      `search_store.py add … --status FAILED --failure CONFIG_INVALID`; propose another.
   c. **Launch trial** `d` (see [§4 Launch a trial](#4-launch-a-trial-the-per-node-eval)) into `$CAMPAIGN/node_<next>-<tag>`,
      then plot, diagnose, and register:
      ```bash
      NID=$(python "$SKILL/helpers/search_store.py" add --campaign-dir "$CAMPAIGN" \
        --parent P --overrides '<d>' --round r --tag <tag> \
        --log-dir "$CAMPAIGN/node_<next>-<tag>" | python3 -c "import sys,json;print(json.load(sys.stdin)['id'])")
      python "$SKILL/helpers/search_store.py" set-result --campaign-dir "$CAMPAIGN" \
        --id "$NID" --from-diagnosis "$CAMPAIGN/node_<next>-<tag>/diagnosis.json"
      ```
      (Add the node *before* launching if you prefer the id for the dir name — either
      order is fine as long as `--log-dir` matches where the trial actually wrote.)
   d. **★ WRITE `$NODE_DIR/REPORT.md` NOW — before launching the next trial.** Do NOT
      defer this to the end of the round. See [§5 Write per-node REPORT.md](#5-write-per-node-reportmd)
      for the full step-by-step. Every OK trial that skips this step breaks the search:
      the next round's proposals depend on each parent node's REPORT.md for evidence.

#### Round checkpoint
4. **Round checkpoint** — before reporting, verify every trial launched this round has
   its REPORT.md. If any is missing, stop and write it NOW:
   ```bash
   for d in "$CAMPAIGN"/node_*; do
     [ -f "$d/REPORT.md" ] || echo "MISSING REPORT.md: $d — write it before proceeding"
   done
   ```
   Do NOT proceed to the round report with missing per-node reports.

#### Report round
5. **Report round:** `search_store.py tree --campaign-dir "$CAMPAIGN"` — show the tree
   + leaderboard to the user. Fill or update the SEARCH_REPORT.md file according to
   [`reference/search-report-template.md`](reference/search-report-template.md).
   If no new node beat the best (all dup/invalid/OOM/slower), you may stop early
   (note the plateau).

### 4. Finish
```bash
python "$SKILL/helpers/search_store.py" best --campaign-dir "$CAMPAIGN"
```
Write `$CAMPAIGN/SEARCH_REPORT.md`: the winning config (its cumulative overrides +
resolved knobs), the leaderboard, per-round progression, and the dead-ends (OOM /
invalid / duplicates). Report the best config and its speedup vs baseline to the user.

### 5. Build Wiki (Self-Learning)

After SEARCH_REPORT.md is written, distill knowledge from the campaign into the
cross-campaign wiki. This creates or updates entries organized by
[env/model/algorithm/cfg/knob-effect] so future searches benefit from this campaign's
experience.

**After the entire search campaign is complete, you write the wiki entries yourself.**
This gives you the flexibility to capture nuanced insights, synthesize across
campaigns, and adapt the structure to what the data actually says — rather than
filling numbers into a rigid template.

1. **Write wiki entries** — read the campaign data and write entries for each
   dimension. Follow the guide at [`reference/wiki-entry-template.md`](reference/wiki-entry-template.md):
   ```bash
   # Read the writing guide first
   # Then read campaign data:
   #   - $CAMPAIGN/SEARCH_REPORT.md
   #   - $CAMPAIGN/nodes.jsonl
   #   - $CAMPAIGN/tree.json
   #   - $CAMPAIGN/baseline_knobs.json
   #   - <best_node_log_dir>/diagnosis.json
   #
   # Write entries directly to the wiki directory:
   #   env/<env>/<algorithm>-<model>.md
   #   model/<model>/<env>-<algorithm>.md
   #   algorithm/<algorithm>/<env>-<model>.md
   #   cfg/<env>-<model>-<algorithm>.md
   #   knob-effect/<category>.md
   ```

2. **Verify consistency** — check for internal consistency, cross-node agreement,
   and memory model sanity:
   ```bash
   python "$WIKI/wiki_index.py" verify \
     --campaign-dir "$CAMPAIGN" --wiki-dir "$WIKI"
   ```

3. **Regenerate INDEX files** — updates the top-level and per-dimension indexes:
   ```bash
   python "$WIKI/wiki_index.py" index --wiki-dir "$WIKI"
   ```

4. **Report to the user** — print a summary of what was added/updated:
   ```
   Wiki updated:
     env/maniskill/ppo-openvlaoft.md  (updated)
     model/openvlaoft/maniskill-ppo.md  (updated)
     algorithm/ppo/maniskill-openvlaoft.md  (new)
     cfg/collocation/maniskill-ppo-openvlaoft.md  (new)
     knob-effect/total_num_envs_scaling.md  (updated)
   INDEX.md regenerated.
   Wiki: $WIKI
   ```

## 7. File index

| Path | Purpose |
|---|---|
| `helpers/search_store.py` | node store (`nodes.jsonl`) + tree (`tree.json`) + CLI (init/add/set-result/dedup/frontier/tree/best) |
| `helpers/preflight.py` | knob domains + divisibility/placement validation (reject invalid configs pre-GPU) |
| `helpers/diagnose.py` | single-run signal extractor → `diagnosis.json` (pipeline balance, generation internals, rank straggler, offload/memory, GPU/CPU) |
| `scripts/gpu_check.sh` | GPU idle preflight (nvidia-smi util < 10%, mem < 2000MiB) |
| `profiler/enable_timeline.sh` | source to activate timeline tracing (sets PYTHONPATH + env vars) |
| `profiler/enable_nvitop.sh` | source to activate nvitop GPU sampling |
| `profiler/plot_timeline.py` | Gantt chart plotter for timeline JSONL (PNG + HTML) |
| `profiler/plot_nvitop.py` | resource curve plotter for nvitop JSONL, also writes `nvitop_summary.log` |
| `profiler/rlinf_timeline/` | import-time monkey-patching engine + samplers (auto-injected via PYTHONPATH) |
| `profiler/sitecustomize.py` | bootstrap auto-import for rlinf_timeline |
| `profiler/timeline_patches.embodied.txt` | patch specs for embodied PPO jobs |
| `profiler/timeline_trace.py` | backward-compatible `append_timeline_event` entry point |
| `reference/knob-schema.md` | tunable knobs, baselines, domains, hard invariants |
| `reference/search-recipe.md` | the beam loop + how to turn a diagnosis into knob deltas |
| `reference/concepts.md` | domain model (per-trajectory metric, generation nesting, offload/OOM memory) |
| `reference/atomic-profile-concepts.md` | domain model for per-component atomic profiling (cost curves, memory modeling, sweep methodology) |
| `reference/atomic-profile-report-template.md` | REPORT.md template for atomic profile sweeps (env/rollout/actor tables) |
| `reference/diagnosis-playbook.md` | signal → cause → fix playbook (Patterns A–I) |
| `reference/report-template.md` | REPORT.md structure and style rules |
| `reference/search-report-template.md` | SEARCH_REPORT.md structure and style rules |
| `wiki/wiki_index.py` | CLI tool for wiki: query/verify/index |
| `wiki/reference/wiki-entry-template.md` | Wiki entry writing guide (agent-driven) |
| `wiki/` | Cross-campaign wiki directory (agent-written entries) |

## 8. Notes

- Booleans in overrides are lowercase (`true`/`false`); overrides are cumulative
  (a child stacks on its parent) — propose only the incremental delta.