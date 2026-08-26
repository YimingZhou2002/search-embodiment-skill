# Wiki Index -- RLinf Embodiment Config Search Experience

Last updated: 2026-08-20

## Dimensions

- [env](env/INDEX.md) -- Per-environment tuning experience
- [model](model/INDEX.md) -- Per-model tuning experience
- [algorithm](algorithm/INDEX.md) -- Per-algorithm tuning experience
- [cfg](cfg/INDEX.md) -- Per-config-pattern tuning experience
- [knob-effect](knob-effect/INDEX.md) -- Cross-campaign knob effect experience

## Usage

When proposing knob deltas, query the wiki for relevant entries:

```bash
python "$SKILL/wiki/wiki_index.py" query --wiki-dir "$SKILL/wiki" \
  --env <env> --model <model> --algorithm <algorithm>
```

