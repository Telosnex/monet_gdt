#!/bin/sh
# Inputs computed by libmonet, the code the app runs (tools/libmonet_tools;
# needs Flutter, since libmonet's Hct imports dart:ui):
#   hct_lut      out/graph/hct_lut.f32: HCT of the graph's 32,768 colors
#   rule_triads  out/teacher_colors/rule_triads.jsonl: rule_v1 triads of the
#                Jev color labels (jev/colors.jsonl), the color model's targets
#   sh scripts/libmonet_tools.sh hct_lut|rule_triads|all
cd "$(dirname "$0")/.." || exit 1
R=$(pwd)
what=${1:-all}
cd tools/libmonet_tools && flutter pub get >/dev/null || exit 1
case $what in hct_lut|all)
  MONET_HCT_LUT="$R/out/graph/hct_lut.f32" flutter test test/hct_lut_test.dart || exit 1;; esac
case $what in rule_triads|all)
  MONET_JEV_COLORS="$R/jev/colors.jsonl" MONET_TRIADS_OUT="$R/out/teacher_colors/rule_triads.jsonl" \
    flutter test test/rule_triads_test.dart || exit 1;; esac
