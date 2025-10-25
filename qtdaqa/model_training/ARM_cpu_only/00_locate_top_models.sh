
ans=$(find training_runs -type f | grep model_checkpoint)

printf "$ans\n" | cut -d "." -f 2,3 | sort -n | head -3 |

while read -r foo
do

   printf "${ans}\n" | grep ${foo}

done
