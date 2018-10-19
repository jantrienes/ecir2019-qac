#!/bin/bash
declare -a communities=("stats" "unix" "askubuntu" "superuser")
# declare -a communities=("stats" "unix" "askubuntu" "superuser" "stackoverflow")
# declare -a communities=("debug")

JOBS=8

for community in "${communities[@]}"
do
  echo "Start retrieval of $community with 'all' querying strategy"
  python qac/simq/simq_retrieval.py "$community" 60stop100body --strategy "all" --body_length 100
  echo "Start feature computation of $community"
  python qac/simq/simq_features.py "$community" 60stop100body --n_jobs "$JOBS"

  echo "Start retrieval of $community with 'constrained' querying strategy"
  python qac/simq/simq_retrieval.py "$community" 60stop0body --strategy "constrained"
  echo "Start feature computation of $community"
  python qac/simq/simq_features.py "$community" 60stop0body --n_jobs "$JOBS"
done
