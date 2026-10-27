#!/bin/sh
# Rebuild the graph for a student (2026-10-26):
# 1 graph with the current color model -> 2 its features (all images, app
#   decode; train_shape_head.py BASE=) -> 3 color model on those features
#   -> 4 graph with that color model (same backbone, same features) ->
#   5 shape head on its features -> 6 add the shape head -> 7 checks.
#   STUDENT=student_matched50k scripts/rebuild_graph.sh  (log per step in out/logs/)
cd "$(dirname "$0")/.." || exit 1
V=.venv/bin/python; export PYTHONWARNINGS=ignore FEAT=student
: "${STUDENT:?set STUDENT}"; export STUDENT
step() { echo "[$(date +%T)] $1"; }
mkdir -p out/logs
[ -f out/graph/hct_lut.f32 ] || sh scripts/libmonet_tools.sh hct_lut || exit 1
[ -f out/teacher_colors/rule_triads.jsonl ] || sh scripts/libmonet_tools.sh rule_triads || exit 1
step "1 graph (old color model)"; $V -u scripts/build_graph.py build > out/logs/rb1_build.log 2>&1 || exit 1
step "2 features"; BASE=out/graph/monet_gdt_w8.onnx $V -u scripts/train_shape_head.py > out/logs/rb2_feats.log 2>&1 || exit 1
step "3 color model"; $V -u scripts/train_colors.py > out/logs/rb3_colors.log 2>&1 || exit 1
step "4 graph (new color model)"; $V -u scripts/build_graph.py build > out/logs/rb4_build.log 2>&1 || exit 1
step "5 shape head"; BASE=out/graph/monet_gdt_w8.onnx $V -u scripts/train_shape_head.py > out/logs/rb5_shapes.log 2>&1 || exit 1
step "6 add shape head"; $V -u scripts/add_shape_head.py out/graph/monet_gdt_w8.onnx out/shapes out/graph/monet_gdt_w8_shapes.onnx > out/logs/rb6_onnx_shapes.log 2>&1 || exit 1
step "7 checks"; N=300 $V -u scripts/build_graph.py check > out/logs/rb7_check.log 2>&1 || exit 1
rm -f out/graph/monet_gdt_fp32.onnx
step "all done"
