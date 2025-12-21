# Value: quick confirmation that the latest run finished, how long it took, and which checkpoints the pipeline considers best—without wading through the rest of the JSON payload.
python monitor_best_model.py --run-dir training_runs2/latest | jq '{run_dir,runtime_estimate,checkpoint_symlinks}'
