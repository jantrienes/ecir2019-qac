#!/bin/bash
declare -a communities=("stats" "unix" "askubuntu" "superuser" "stackoverflow")

for community in "${communities[@]}"
do
  echo "Evaluate $community"
  python qac/evaluation/evaluate_community.py "$community"
done
