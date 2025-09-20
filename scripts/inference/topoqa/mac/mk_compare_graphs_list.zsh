
DOCKGROUND_GRAPHS="logs/output/work/Dockground"
MAF2_GRAPHS="logs/output/work/MAF2"

# location of script
script_dir="$(cd -- "$(dirname -- "$0")" && pwd -P)"

echo ${script_dir}

# Dockground graphs
find ${script_dir}/${DOCKGROUND_GRAPHS} -type f |
grep "/graph/"

echo

# MAF2 graphs
find ${script_dir}/${MAF2_GRAPHS} -type f |
grep "/graph/"
