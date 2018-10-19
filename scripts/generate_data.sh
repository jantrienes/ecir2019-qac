#!/bin/bash
community="unix"
xml_dir=data/"$community".stackexchange.com/
out_dir=data

python qac/dataset/load_data.py "$community" "$xml_dir"
python qac/dataset/annotate_data.py "$community"
python qac/dataset/create_indexes.py
python qac/dataset/dump_community.py "$community" "$out_dir"/labeled/
python qac/dataset/dump_clarq.py unix "$out_dir"/clarq

python qac/dataset/data_analysis.py "$community" "$out_dir/labeled/$community.csv"
