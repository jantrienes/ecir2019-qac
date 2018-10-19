#!/bin/bash
# declare -a communities=("stats" "unix" "askubuntu" "superuser")
# declare -a communities=("stats" "unix" "askubuntu" "superuser" "stackoverflow")
declare -a communities=("debug")

simq_run=60stop100body
# simq_run=60stop0body

for community in "${communities[@]}"
do
  echo "Start simq classification for $community"
  python qac/simq/simq_majority.py "$community" "simq_${simq_run}_majority" "$simq_run"

  python qac/simq/simq_threshold_classifier.py "$community" "$simq_run" feat_unclear_global_cos
  python qac/simq/simq_threshold_classifier.py "$community" "$simq_run" feat_unclear_individual_cos
  python qac/simq/simq_threshold_classifier.py "$community" "$simq_run" feat_unclear_individual_cos_weighted
  python qac/simq/simq_threshold_classifier.py "$community" "$simq_run" feat_post_readability

  python qac/simq/simq_ml.py "$community" "simq_${simq_run}_ml_1" "$simq_run" --feature_group 1
  python qac/simq/simq_ml.py "$community" "simq_${simq_run}_ml_2" "$simq_run" --feature_group 2
  python qac/simq/simq_ml.py "$community" "simq_${simq_run}_ml_3" "$simq_run" --feature_group 3
  python qac/simq/simq_ml.py "$community" "simq_${simq_run}_ml_12" "$simq_run" --feature_group 1 2
  python qac/simq/simq_ml.py "$community" "simq_${simq_run}_ml_all" "$simq_run" --feature_group all
done
