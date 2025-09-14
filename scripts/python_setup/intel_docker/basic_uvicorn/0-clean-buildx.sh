# 0) Make sure we’re using the healthy Desktop builder
docker buildx use desktop-linux

# 1) Remove stopped/errored builders (safe to delete)
for b in gracious_euler modest_kapitsa naughty_hoover colima; do
  docker buildx rm -f "$b" || true
done

# 2) Prune unused BuildKit cache on the active builders
for b in desktop-linux default; do
  docker buildx prune --builder "$b" -a -f || true
done

# 3) Clean up any orphaned buildkit containers (rare but harmless)
docker ps -a --filter "name=buildx_buildkit" -q | xargs -r docker rm -f

# 4) (Optional) Broader Docker cleanup (be careful: removes dangling images/containers/networks/volumes)
# docker system prune -a --volumes -f

