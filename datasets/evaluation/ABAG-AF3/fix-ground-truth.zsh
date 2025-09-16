# repair label.txt
# MODEL NAME, PDB model file, DockQ_Wave, QS_Best

typeset -u upper_target
output_file="label_info.csv"

printf "Target,Model,DockQ_Wave,QS_Best\n" > ${output_file}

grep -v ^MODEL label.txt |
while read -r lines
do
    upper_target=$(printf "${lines}\n" |
                   cut -d "_" -f 2
                   )

    pdb_file=$(printf "${lines}\n" |
               cut -d " " -f 1
              )

    dockq_wave=$(printf "${lines}\n" |
                 cut -d " " -f 2
                ) 

    qs_best=$(printf "${lines}\n" |
              cut -d " " -f 3
             ) 

    printf "${upper_target},${pdb_file},${dockq_wave},${qs_best}\n"

done >> ${output_file}