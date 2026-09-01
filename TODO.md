# TODO.md — RLinf Embodiment Config Beam Search Agent Workflow

> **Goal:** Find the best config (lowest `step_time_per_traj_s`) for an RLinf embodiment training job via beam search.
>
> **Method:** Beam search (width 2, branching 2), up to 10 rounds, 4 trials/round. Each trial is a short profiled run.
>
> **Config:** `<CONFIG>` (default: `maniskill_ppo_openvla`) | **STEPS:** `2` | **GPUs:** all 8

---

## Phase 0: Campaign Setup

- [ ] **0.1** Pick campaign dir: `CAMPAIGN=$RLINF_ROOT/logs/search_$(date +%Y%m%d-%H%M%S)` and `mkdir -p` it.
- [ ] **0.2** Confirm config name, ROUNDS, BEAM, BRANCH, STEPS with user.
- [ ] **0.3** Verify `gpu_check.sh` passes (GPUs idle).

## Phase 1: Cold Start — Baseline

- [ ] **1.1** Launch baseline trial (no overrides) → `$CAMPAIGN/node_0-baseline` (see SKILL.md §4).
- [ ] **1.2** Wait for trial to complete.
- [ ] **1.3** Plot timeline + nvitop (PNG + HTML) for `node_0-baseline`.
- [ ] **1.4** Run `diagnose.py` on `node_0-baseline` → `diagnosis.json`.
- [ ] **1.5** Register root node via `search_store.py init --campaign-dir "$CAMPAIGN" --log-dir "$CAMPAIGN/node_0-baseline" --tag baseline`.
- [ ] **1.6** Set baseline result via `search_store.py set-result --campaign-dir "$CAMPAIGN" --id 0 --from-diagnosis "$CAMPAIGN/node_0-baseline/diagnosis.json"`.
- [ ] **1.7** **Write `$CAMPAIGN/node_0-baseline/REPORT.md`** (see SKILL.md §5).

## Phase 2: Atomic Modeling — Per-Component Cost Curves

- [ ] **2.1** Run worker-level atomic-modeling sweep (see SKILL.md §3):
  ```bash
  bash "$SKILL/scripts/gpu_check.sh" && \
  cd "$RLINF_ROOT" && source "$VENV/bin/activate" && \
  RLINF_BENCH_OUT="$CAMPAIGN/atomic_model" RLINF_BENCH_ENV_MAX=64 \
    bash examples/benchmark/run_sweep.sh <CONFIG>
  ```
- [ ] **2.2** Verify sweep completed: `ls "$CAMPAIGN/atomic_model/bench_msgs/sweep_"*.{json,csv}`.
- [ ] **2.3** Read and understand the 3 cost curves (env, rollout, actor) — these drive all future proposals.

## Phase 3: Beam Search Rounds

> For each round `r` in `1..ROUNDS` (max 10), execute the following:

### Round R — Frontier & Proposals

- [ ] **R.1** **Frontier:** select top-2 expandable nodes:
  ```bash
  python "$SKILL/helpers/search_store.py" frontier --campaign-dir "$CAMPAIGN" --k 2 --max-children 2 --composite
  ```
- [ ] **R.2** **Query wiki** for relevant cross-campaign knowledge:
  ```bash
  python "$WIKI/wiki_index.py" query --wiki-dir "$WIKI" --env <ENV> --model <MODEL> --algorithm <ALGO>
  ```
- [ ] **R.3** For each frontier node, read its `diagnosis.json` and `REPORT.md`.
- [ ] **R.4** **Propose** `BRANCH=2` knob deltas per frontier node (total up to 4), using:
  - `reference/search-recipe.md` (priority mapping)
  - `reference/knob-schema.md` (legal domains + invariants)
  - `reference/diagnosis-playbook.md` (signal → cause → fix)
  - Atomic modeling curves (see §3)
  - Wiki knowledge (prioritize confirmed directions, avoid dead ends)
  - **Early rounds:** single-knob deltas (breadth). **Later rounds:** stacked deltas (depth).

### Round R — Per-Proposal (4 sequential trials)

- [ ] **R.5** For each proposal `d` on parent `P`:
  - [ ] **R.5a** **Dedup:** `search_store.py dedup --campaign-dir "$CAMPAIGN" --parent P --overrides '<d>'`. If duplicate, skip (reuse known objective) and propose a different delta.
  - [ ] **R.5b** **Preflight:** `preflight.py --config <config.yaml> --overrides '<d>'`. If invalid, record FAILED node and re-propose.
  - [ ] **R.5c** **Launch trial** (see SKILL.md §4) into `$CAMPAIGN/node_<N>-<tag>` with `run_in_background: true`.
  - [ ] **R.5d** Wait for trial to complete.
  - [ ] **R.5e** Plot timeline + nvitop (PNG + HTML) for the node.
  - [ ] **R.5f** Run `diagnose.py` on the node → `diagnosis.json`.
  - [ ] **R.5g** Register node via `search_store.py add` + `set-result`.
  - [ ] **R.5h** **★ Write `$NODE_DIR/REPORT.md` NOW** (before launching next trial!). See SKILL.md §5.

### Round R — Checkpoint & Report

- [ ] **R.6** **Round checkpoint:** verify every trial launched this round has its `REPORT.md`:
  ```bash
  for d in "$CAMPAIGN"/node_*; do
    [ -f "$d/REPORT.md" ] || echo "MISSING REPORT.md: $d"
  done
  ```
- [ ] **R.7** **Report round:** show tree + leaderboard:
  ```bash
  python "$SKILL/helpers/search_store.py" tree --campaign-dir "$CAMPAIGN"
  ```
- [ ] **R.8** Update `$CAMPAIGN/SEARCH_REPORT.md` (see `reference/search-report-template.md`).
- [ ] **R.9** Run exhaustion check:
  ```bash
  python "$SKILL/helpers/search_store.py" exhaustion-check \
    --campaign-dir "$CAMPAIGN" --k 2 --max-children 2 \
    --min-headroom-gib 5.0 --min-rounds 5
  ```
  If `should_stop: true`, break early. Otherwise continue to next round.

## Phase 4: Finish — Final Report

- [ ] **4.1** Get best node:
  ```bash
  python "$SKILL/helpers/search_store.py" best --campaign-dir "$CAMPAIGN"
  ```
- [ ] **4.2** Finalize `$CAMPAIGN/SEARCH_REPORT.md` with:
  - Winner (best per-traj time + delta vs baseline)
  - Winning config (cumulative overrides + resolved knobs)
  - Leaderboard
  - Search tree
  - Per-round progression
  - Knob sensitivity map
  - Key insights
  - Conclusion
- [ ] **4.3** Report best config + speedup vs baseline to user.

## Phase 5: Build Wiki (Self-Learning)

- [ ] **5.1** Read campaign data: `SEARCH_REPORT.md`, `nodes.jsonl`, `tree.json`, `baseline_knobs.json`, best node's `diagnosis.json`.
- [ ] **5.2** Write wiki entries for each dimension:
  - `env/<env>/<algorithm>-<model>.md`
  - `model/<model>/<env>-<algorithm>.md`
  - `algorithm/<algorithm>/<env>-<model>.md`
  - `cfg/<pattern>/<env>-<model>-<algorithm>.md`
  - `knob-effect/<category>.md`
- [ ] **5.3** Verify consistency:
  ```bash
  python "$WIKI/wiki_index.py" verify --campaign-dir "$CAMPAIGN" --wiki-dir "$WIKI"
  ```
- [ ] **5.4** Regenerate INDEX files:
  ```bash
  python "$WIKI/wiki_index.py" index --wiki-dir "$WIKI"
  ```
- [ ] **5.5** Report wiki updates to user.

---

## Reference Cheat Sheet

| File | Purpose |
|------|---------|
| `SKILL.md` | Master workflow doc |
| `reference/search-recipe.md` | Beam loop + proposal logic |
| `reference/knob-schema.md` | Tunable knobs + invariants |
| `reference/diagnosis-playbook.md` | Patterns A–I (signal → cause → fix) |
| `reference/concepts.md` | Domain model (per-traj metric, pipeline, memory, offload) |
| `reference/atomic-profile-concepts.md` | Atomic profiling methodology |
| `reference/report-template.md` | Per-node REPORT.md template |
| `reference/search-report-template.md` | Campaign SEARCH_REPORT.md template |
| `reference/wiki-entry-template.md` | Wiki entry writing guide |
| `wiki/INDEX.md` | Cross-campaign wiki index |