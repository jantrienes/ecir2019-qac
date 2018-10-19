#!/bin/bash
declare -a communities=("stats" "unix" "askubuntu" "superuser")
# declare -a communities=("debug")

for community in "${communities[@]}"
do
  echo "Start $community"
  python qac/baseline/baseline_lr_cv.py "$community" baseline_lr --n_jobs 18
  python qac/baseline/baseline_lr_fixed_n.py "$community" baseline_lr_1ngram_c1 --ngram_range 1
  python qac/baseline/baseline_lr_fixed_n.py "$community" baseline_lr_3ngram_c1 --ngram_range 3
done
