# Value: a fast sanity check that symlinks point to the right files and match the leaderboard/monitor output, handy before packaging or resuming from a given checkpoint.
python -m train_cli summarise --run-dir training_runs2/sched_boost_seed222_2025-11-05_23-27-09_phase2_seed888 | jq '{checkpoint_symlinks}'
