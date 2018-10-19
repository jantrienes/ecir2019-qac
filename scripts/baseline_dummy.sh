#!/bin/bash
declare -a communities=("stats" "unix" "askubuntu" "superuser" "stackoverflow")
# declare -a communities=("debug")

for community in "${communities[@]}"
do
  echo "Start $community"
  python qac/baseline/baseline_dummy.py "$community" baseline_majority --strategy most_frequent
  python qac/baseline/baseline_dummy.py "$community" baseline_random --strategy uniform
done
