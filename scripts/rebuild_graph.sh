#!/bin/sh
# Build the graph for a student (ADR D5 step 6, D12). Log per step in out/logs/.
#   STUDENT=student_matched50k sh scripts/rebuild_graph.sh
# 1-4: the base graph (scores, feature, shapes; placeholder colors).
# 5-9: the color step: data, labels from monet_ui, color head, final graph.
# Needs Flutter (tools/v2_labels; monet_ui comes from pub.dev).
cd "$(dirname "$0")/.." || exit 1
V=.venv/bin/python; export PYTHONWARNINGS=ignore
: "${STUDENT:?set STUDENT}"; export STUDENT
step() { echo "[$(date +%T)] $1"; }
mkdir -p out/logs out/v2
D="$PWD/out/v2"
step "1 base graph"; $V -u scripts/build_graph.py build > out/logs/rb1_build.log 2>&1 || exit 1
step "2 shape head"; BASE=out/graph/monet_gdt_w8.onnx $V -u scripts/train_shape_head.py > out/logs/rb2_shapes.log 2>&1 || exit 1
step "3 add shape head"; $V -u scripts/add_shape_head.py out/graph/monet_gdt_w8.onnx out/shapes out/graph/monet_gdt_w8_shapes.onnx > out/logs/rb3_onnx_shapes.log 2>&1 || exit 1
step "4 base checks"; N=300 $V -u scripts/build_graph.py check > out/logs/rb4_check.log 2>&1 || exit 1
rm -f out/graph/monet_gdt_fp32.onnx
step "5 color data"; $V -u scripts/v2_prep.py > out/logs/rb5_prep.log 2>&1 || exit 1
step "6 color labels and tables (monet_ui)"; rm -f out/v2/labels_*.jsonl
(cd tools/v2_labels && MONET_V2_DIR="$D" flutter test test/v2_labels_test.dart && MONET_V2_DIR="$D" flutter test test/v2_tables_test.dart) > out/logs/rb6_labels.log 2>&1 || exit 1
step "7 color head"; $V -u scripts/train_colors_v2.py > out/logs/rb7_colors.log 2>&1 || exit 1
step "8 graph"; $V -u scripts/build_graph_v2.py build > out/logs/rb8_graph.log 2>&1 || exit 1
step "9 checks (PyTorch, base outputs, role order)"; N=300 $V -u scripts/build_graph_v2.py check > out/logs/rb9_check.log 2>&1 || exit 1
(cd tools/v2_labels && MONET_V2_IN="$D/check_triads.jsonl" MONET_V2_OUT="$D/check_order.jsonl" flutter test test/v2_order_test.dart) > out/logs/rb9_order.log 2>&1 || exit 1
$V -c "
import json, sys
L = lambda f: {r['id']: r['triad'] for r in map(json.loads, open(f))}
a, b = L('out/v2/check_triads.jsonl'), L('out/v2/check_order.jsonl')
bad = [i for i in a if a[i] != b[i]]
print(f'role order equals MonetGdtRule20261030 on {len(a) - len(bad)}/{len(a)}')
sys.exit(1 if bad or len(a) != len(b) else 0)
" || exit 1
step "all done: out/v2/monet_gdt_v2.onnx"
