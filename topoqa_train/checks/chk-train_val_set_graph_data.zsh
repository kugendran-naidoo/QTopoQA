
grep -v '^MODEL' ../train.csv |
while IFS=',' read -r target1 dockq1 capri1 _
do
    ls -1 ../graph_data/${target1}.pt >/dev/null 2>&1 ||
    printf 'missing %s\n' "$target1"

done

grep -v '^MODEL' ../val.csv |
while IFS=',' read -r target1 dockq1 capri1 _
do
    ls -1 ../graph_data/${target1}.pt >/dev/null 2>&1 ||
    printf 'missing %s\n' "$target1"

done
