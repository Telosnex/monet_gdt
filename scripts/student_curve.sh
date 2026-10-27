#!/bin/sh
# Student learning curve: crop students (AUG=1, recipe of out/student_aug)
# on 1/4 and 1/2 of the train split (nested, sha256 order; full teacher as
# target). Full set = out/student_aug (existing).
#   scripts/student_curve.sh   (log: out/logs/student_curve.log)
cd "$(dirname "$0")/.." || exit 1
V=.venv/bin/python
for n in 6308 12615; do
  [ -f out/curves/student_aug_$n/student.pt ] || AUG=1 N=$n EPOCHS=20 PATIENCE=5 PYTHONWARNINGS=ignore $V -u scripts/train_student.py || exit 1
done
echo "student curve: all done"
