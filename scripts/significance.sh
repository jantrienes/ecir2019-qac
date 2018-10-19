#!/bin/bash
declare -a communities=("stats" "unix" "askubuntu" "superuser" "stackoverflow")

JOBS=8

for community in "${communities[@]}"
do
  echo "Test significance for $community..."
  python qac/evaluation/significance_testing.py "$community" --n_jobs "$JOBS"
done
