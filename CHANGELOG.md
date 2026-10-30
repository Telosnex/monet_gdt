# Changelog

Each version is its release date, Y.M.D without zero padding. A second
release on the same day adds `.N`, for example `2026.10.30.1`. The ONNX
`model_version` is YYYYMMDD × 100 + N. Design and evidence:
`docs/ADR_train_monet_gdt.md` (D12, appendix B).

## 2026.10.30

The colors change. The inputs, the outputs, the mood scores, the feature
and the shapes do not change.

- Color labels come from `MonetGdtRule20261030` in monet_ui, not from Jev.
  The rule chooses colorful colors before large areas. It needs 128
  clusters from `QuantizerCelebi`.
- A new color head picks P, S and T from the full color list of the image.
  It also gets the share of the image within ΔE 10 of each color.
- Two fixed rules follow the head. S and T are never within ΔE 10 of an
  earlier pick. S is the nearer step from P, and T is the larger step, as
  in `MonetGdtRule20261030`.
- The old color head and rule_v1 are removed from the graph.
- Evidence, from blind votes and Jev checks on held-out images:
  - Against 2026.10.27, the first version of the new head won 34 to 8.
    The released head won 35 to 15 against that first version.
  - Jev found a missing colorful color on 3 of 512 images, against 43
    for 2026.10.27. Jev found a small speck color on 12, against 0.
- The file is 5.55 MB, the same as before. On one CPU core of a desktop
  Mac, a run takes 15.3 ms, against 14.6 ms.

## 2026.10.27

The first release. The graph metadata gives its version as `1.0.0`.
