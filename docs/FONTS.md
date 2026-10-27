# Fonts: from the graph's mood scores to three fonts

The graph gives 10 raw mood scores (`scores`). The app turns them into a
headline, body and label font from Google Fonts. This repository builds
the graph and the mood map; the per-font data is built separately and is
not part of this repository.

## In the app

1. **Percentiles.** `monet_gdt_mood.json` (`scripts/export_app.py`) maps
   each raw score to a percentile among a wallpaper population: the
   teacher's quantile map (raw -> Jev's scale) composed with the
   population's values (`lib/mood_map.py`).
2. **Headline.** Families with headline score >= 0.5, not monospace; the
   one whose 10 mood percentiles are closest (mean squared difference).
3. **Body.** Role score >= 0.5, not monospace, not all-caps, not the
   headline font; shape similarity to the headline in [0.5, 0.8] (if none,
   those within 0.05 of the closest); then the closest mood.
4. **Label.** As body; all-caps allowed; not the body font.
5. Ties: the first name alphabetically. Shuffle: a random headline; body
   and label then match the headline's own mood.

The [0.5, 0.8] band won blind A/Bs: 12-4 over [0.2, 0.5], which had
beaten [-0.1, 0.2] 11-6.

## The font data

Built outside this repository. Per Google Fonts family:

| Field | From |
|---|---|
| `category` | Google Fonts catalog |
| `mood` (10 percentiles among fonts) | Jev, one call per family, on a Flutter-rendered specimen (3 lines, 60 px) |
| `headline`, `body`, `label` | Jev: "Is this a well-designed typeface for ...?" |
| `shape` (128 numbers, 8-bit) | DINOv3 ViT-S/16 patch mean on a "Laseg / dhum / Hloiv" specimen, centered, PCA |
| `caps`, width | measured from the rendered pixels |

Families need a-z, A-Z, 0-9, no color glyphs, and pass Jev's validity
score and an exclusion list. 1,808 families (2026-10-26).

## In this repository

- `fonts/mood_font_model.json`: an older font model (410 families, mood
  vectors from Jev). Used only for the "font top 5%" metric while
  training (`scripts/train_teacher.py`): whether the font the model's mood
  picks is among the top 5% for Jev's mood. A proxy to compare models,
  not the app's pick (no role filter, 410 families, not 1,808).
- `fonts/wall_population.json` (`scripts/wall_population.py`): the
  wallpaper population for step 1.
