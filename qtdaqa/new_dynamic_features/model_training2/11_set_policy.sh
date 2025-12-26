python tools/stats_split_helper.py \
  --train ./data.train.csv \
  --val ./data.validate.csv \
  --eval /Volumes/PData/Data/Dev/Github/Repos/phd3/qtopo/QTopoQA/datasets/training/adjusted/Dockground_MAF2/label_info_Dockground_MAF2.csv \
  --out ./stats_split_helper.tuning.csv \
  --stats-out ./stats_split_helper.tuning_stats.json \
  --tuning-fraction 0.25 \
  --min-group-size 2 \
  --singleton-policy replace
# other policies
# --singleton-policy replace
# --singleton-policy hybrid
