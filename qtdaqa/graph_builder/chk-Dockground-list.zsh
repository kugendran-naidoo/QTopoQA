
skip_list=$(grep '\[skip\]' logs/graph_builder.2025-09-21_14:49.log | 
            sort | 
            cut -d "=" -f 3 | 
            cut -d "." -f 1 |
            grep -vE "_u1|_u2|_p1|_p2"
           )

Dockground_list=$(find ../../scripts/inference/topoqa/mac/logs/output/work/Dockground -type f |
           grep /graph/
          )

results=$(printf "${skip_list}\n" |
          while read -r skipped
          do

            echo ${Dockground_list} |
            grep "${skipped}.pt" &&
            { printf "${skipped} found\n"
            } || 
            printf "${skipped} skipped\n"

          done 
        )

total=$(echo ${skip_list} | grep -c .)
found=$(printf "${results}\n" | grep -c found)
skipped=$(printf "${results}\n" | grep -c skipped)

printf "total skipped = ${total}\n"
printf "actually skipped = ${skipped}\n"
printf "actually found = ${found}\n"