# Experiments

The studies behind the design decisions in
[docs/ADR_train_monet_gdt.md](../docs/ADR_train_monet_gdt.md). None is
needed to build the model. Each folder's `results/` holds the evidence
(votes, Jev answers, summaries); `data/` and `out/` are scratch and not
committed.

| Folder | Question | Outcome |
|---|---|---|
| `synthetic_ab/` | Do MONET synthetic images help the mood heads? | Real + synthetic beats real only on wallpapers (ADR D1) |
| `genre_ab/` | Do genre-targeted synthetic images (dark, game, anime) help further? | Yes; the training mix is 1/2 real, 1/4 generic, 1/4 genre (D1) |
| `museum_ab/` | Does CC0 museum art help? | No; it hurt every mix, dropped (D1) |
| `color_options_ab/` | How many candidate colors should Jev choose from? | At most 12 merged colors (D10) |
| `color_prompt_ab/` | Which color prompt and pick rule? | rule_v1, used by the graph (D10) |

Several of these read images and pages from private repositories
(`TELOSNEX`, `LIBMONET_GDT` environment variables), so they are a record
of what was run rather than something to re-run.
