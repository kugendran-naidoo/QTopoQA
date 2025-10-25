git fetch origin
git rev-list --left-right --count HEAD...origin/main
git log --oneline HEAD..origin/main
git diff --name-status HEAD..origin/main
git diff --stat HEAD..origin/main
