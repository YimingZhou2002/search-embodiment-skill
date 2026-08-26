# Wiki Entry Writing Guide — RLinf embodiment config search experience

Wiki entries preserve knowledge from completed campaigns so future searches can
benefit from past experience. **you — the agent — write these
entries yourself**, reading the raw campaign data and applying your judgment.

**Your goal:** Write entries that are *insightful, specific, and actionable*.
A future agent should be able to read an entry and immediately know what knobs
to prioritize, what to avoid, and why.

---

## Data sources

Each campaign directory contains:

| File | What it has |
|------|-------------|
| `SEARCH_REPORT.md` | Human-written summary of the campaign: best config, leaderboard, round progression, dead ends, key findings. **Start here.** |
| `nodes.jsonl` | Every trial node: id, parent, overrides, status (OK/FAILED), objective (s/traj), log_dir, resolved_config, tag |
| `tree.json` | Search tree structure: parent-child relationships, leaderboard |
| `baseline_knobs.json` | Baseline config knobs (before any overrides) |
| `node_<id>-<tag>/diagnosis.json` | Per-node: pipeline_split, memory, efficiency, oom_risk, offload_cost |

Read these files before writing. The `diagnosis.json` is especially important —
it's where the detailed bottleneck analysis lives.

---

## File naming conventions

Entries are organized into 5 dimensions. Follow these paths exactly so the wiki
INDEX and query tools work correctly.

| Dimension | Path pattern | When to write |
|-----------|-------------|---------------|
| **env** | `env/<env>/<algorithm>-<model>.md` | Always |
| **model** | `model/<model>/<env>-<algorithm>.md` | Skip if model is "unknown" |
| **algorithm** | `algorithm/<algorithm>/<env>-<model>.md` | Skip if algorithm is "unknown" |
| **cfg** | `cfg/<pattern>/<env>-<model>-<algorithm>.md` | Always |
| **knob-effect** | `knob-effect/<category>.md` | One per knob-effect category explored |

The `env` entry is the most comprehensive. The `model`, `algorithm`, and `cfg`
entries can be shorter, focusing on insights specific to that dimension.

---

## Frontmatter format

Every entry starts with YAML frontmatter. This is parsed by the INDEX generator
and query tool, so it must be exactly as specified:

```yaml
---
dimensions:
  env: <env-name>
  model: <model-name>
  algorithm: <algorithm-name>
  cfg_pattern: <cfg-pattern-name>
campaign: <campaign-dir-name>
created: <YYYY-MM-DD>
updated: <YYYY-MM-DD>
confidence: <high|medium|low>
baseline_speed: <baseline-obj> s/traj
best_speed: <best-obj> s/traj
speedup: "<delta-pct>%"
num_nodes: <N>
---
```

### Confidence guidelines

- **high**: ≥3 OK nodes with consistent results, clear bottleneck diagnosis
- **medium**: 1–2 OK nodes, or some variance in results
- **low**: Single trial, high variance, or unclear diagnosis

*State the reason for your confidence rating in the entry body, not just in the
frontmatter.*

---

## What to write — by dimension

### env entry (the most important one)

The env entry is the main wiki page for a given (env, algorithm, model)
combination. It should tell a future agent:

1. **What was the bottleneck?** Generation-bound? Training-bound? Offload-tax?
   CPU-bound? Cite specific numbers from `diagnosis.json` pipeline_split.
2. **Which knobs worked — and which didn't?** For each knob explored: what
   range, what direction, what magnitude of effect. Be specific: "increasing
   `total_num_envs` from 256→512 reduced step time by 0.06s (11%)" not "tne
   helped a bit."
3. **Memory profile.** Peak GPU memory, OOM risk, whether offload is needed.
   If the campaign pushed close to the OOM wall, say what the ceiling is.
4. **Dead ends.** What configurations failed and why? Be specific enough that
   a future agent can avoid them without re-running.
5. **Cross-campaign pattern.** If this is a merge of multiple campaigns, how
   do the results compare? Do they agree or conflict?

### model entry

Focus on insights specific to the model architecture. For example:
- How does this model respond to micro_batch_size scaling?
- Does it have high memory pressure compared to other models?
- Is it particularly sensitive to pipeline_stage_num?

### algorithm entry

Focus on algorithm-specific insights:
- Does PPO benefit more from tne scaling than GRPO?
- Is the algorithm compute-bound or memory-bound?
- Does it respond well to gradient checkpointing?

### cfg entry

Focus on the config pattern (collocation, tne_scaling, offload, etc.):
- What was the pattern, and why was it tried?
- What was the observed effect?
- Under what conditions does this pattern work best?

### knob-effect entry

This is the most cross-campaign dimension. When merging data from multiple
campaigns:
- Do NOT just append rows to a table. **Compare and contrast.**
- If the same knob had different effects in different campaigns, explore why.
- Look for patterns: "enable_offload=false helps when env is the bottleneck,
  but hurts when generation is the bottleneck."

---

## Writing principles

### 1. Lead with the bottleneck

The most important insight is **what's slowing things down**. State the dominant
bottleneck first, then explain how the knobs addressed it.

> **Good:** "Generation is the dominant bottleneck (99.8% of step time, actor
> GPUs idle 94.9% of generation time). Increasing `total_num_envs` from 256→512
> improved amortization and reduced step time by 11.3%."
>
> **Bad:** "We tried several knobs. tne helped. offload hurt. The best config
> was 11.3% faster."

### 2. Be specific about numbers

Every claim should cite a specific value. Avoid vague statements.

> **Good:** "`actor.micro_batch_size` from 40→80: magnitude 0.069s/traj (small),
> direction negative (higher = worse). Best value: 40."
>
> **Bad:** "micro_batch_size was slightly better at lower values."

### 3. Explain the "why"

Don't just report *what* happened — explain *why* it happened.

> **Good:** "Increasing `total_num_envs` from 256→512 reduced generation overhead
> because the pipeline had more work to amortize across, reducing idle time per
> trajectory. The effect plateaued at 512 (further increase to 768 showed
> diminishing returns)."
>
> **Bad:** "tne=512 was best."

### 4. Be honest about confidence

If you only have one data point, say so. If the results are noisy, say so.

> **Good:** "This finding is based on a single trial (low confidence). The
> 0.06s improvement is within the noise floor of the measurement, so it should
> be validated with a longer run."
>
> **Bad:** "tne scaling gives 11% speedup." (when it's from one trial)

### 5. Write for the future agent

Imagine you're giving advice to yourself on the next campaign. What would you
want to know?

- What **not** to try (dead ends)
- What **conditions** matter (e.g., "collocation only helps when memory headroom
  > 20%")
- What **combo** worked best (e.g., "tne=512 + offload=false + mbs=40")

### 6. Cross-reference other entries

Use `[[dim/name]]` syntax to link to related entries:
- `[[env/<env>/<algo>-<model>]]`
- `[[model/<model>/<env>-<algo>]]`
- `[[algorithm/<algo>/<env>-<model>]]`
- `[[knob-effect/<category>]]`

### 7. Include raw data as JSON

End each entry with a `## Raw Data Summary` block containing a ````json` block
with the key data. This is useful for programmatic access without parsing
markdown. Include at minimum:

```json
{
  "knob_effects": {
    "<knob.path>": {
      "direction": "<positive|negative|none>",
      "magnitude": <float>,
      "range": ["<min>", "<max>"],
      "confidence": "<high|medium|low>"
    }
  },
  "memory": {
    "max_used_gib": <float>,
    "total_gib": <int>,
    "max_used_pct": <float>,
    "oom_risk": "<safe|high|unknown>"
  },
  "dominant_bottleneck": "<pattern>",
  "bottleneck_detail": "<key-metric>"
}
```

---

## Merging with existing entries

When a campaign is added to a wiki dimension that already has entries,
**do not** simply append data. **You are the agent — synthesize.**

1. Read the existing entry first.
2. Compare the new campaign's findings with the old ones.
3. If they agree, strengthen the confidence and add a note: "Confirmed by
   campaign <X>."
4. If they disagree, note the conditions that differ (different env? different
   GPU count? different model version?) and don't override.
5. Update the frontmatter: preserve `created`, update `updated`, adjust
   `confidence` based on the combined evidence.

This is the key improvement over the old `wiki_store.py`, which just appended
rows mechanically. **You are replacing that rigid logic with your judgment.**

---

## Structure suggestions (not requirements)

These are recommendations, not a rigid template. Adapt as the data demands.

```
---
<frontmatter>
---

# <Title>

## Key Findings

2–4 findings, ordered by impact. Each finding:
- States the bottleneck or knob effect
- Cites specific numbers
- Explains the "why"

## Knob Sensitivity Map

| Knob | Range Explored | Direction | Magnitude | Best Value |
|------|---------------|-----------|-----------|------------|

## Memory & Bottleneck Profile

Peak memory, OOM risk, bottleneck pattern, and detail.

## Known Dead Ends

Specific configs that failed, and why. Include enough detail to avoid re-running.

## Cross-References

[[links]] to related entries.

## Raw Data Summary

```json
{ ... }
```
```

---

## Anti-patterns

- ❌ **Vague findings.** "tne helped" → say how much and under what conditions.
- ❌ **No numbers.** Every claim needs a specific value.
- ❌ **Overconfident from one trial.** Label low-confidence findings clearly.
- ❌ **Ignoring conditions.** A finding without env/model/algorithm context is
  not actionable.
- ❌ **Mechanical merging.** Don't just append new data to old — synthesize.
- ❌ **No dead ends.** Failed trials are valuable knowledge. Record them.
- ❌ **Copying the raw data format as the summary.** The JSON block is for
  machines; the markdown text is for humans. Don't let the JSON be the only
  documentation.