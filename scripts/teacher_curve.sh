#!/bin/sh
# Teacher learning curve: fine-tune on 1/8, 1/4, 1/2 of the train split
# (nested, sha256 order), then score D8. Full set = out/teacher (existing).
#   scripts/teacher_curve.sh   (log: out/logs/teacher_curve.log)
cd "$(dirname "$0")/.." || exit 1
V=.venv/bin/python
for n in 3154 6308 12615; do
  [ -f out/curves/teacher_$n/teacher.pt ] || N=$n PYTHONWARNINGS=ignore $V -u scripts/train_teacher.py || exit 1
  PYTHONWARNINGS=ignore $V -u scripts/score_d8.py out/curves/teacher_$n || exit 1
done
echo "teacher curve: all done"
