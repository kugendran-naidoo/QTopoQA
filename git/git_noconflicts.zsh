# git pull --ff-only
# git add <your files>          # or: git add -A
# git commit -m "Your message"
# git push

git fetch origin
git rebase origin/main
printf "git add … \n"
printf "git push --force-with-lease   # safe since you’re solo\n"

