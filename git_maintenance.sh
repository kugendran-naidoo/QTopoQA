# One-off cleanup
git gc --prune=now --aggressive
git repack -Ad --write-bitmap-index
git commit-graph write --reachable --changed-paths

# Background maintenance (Git ≥ 2.30)
git maintenance start
git maintenance run --task=gc --task=commit-graph --task=prefetch

