#!/bin/bash
# mkdir dats
if [ ! -d "dats" ]; then
  mkdir dats
fi

version=num_primos_paralelo_normal
version=num_primos_fila_concorrente_normal
rm ../plot/$version.dat
for ((i=1; i<=12; i++))
do
    #getting the execution times
    awk '/Tempo de execução/ { print $5 }' $version-$i.log >> dats/$version-$i.dat

    #calculating the avgs and stdevs
    results=$(awk '{for(i=1;i<=NF;i++) {sum[i] += $i; sumsq[i] += ($i)^2}} 
          END {for (i=1;i<=NF;i++) {
          printf " %f %f \n", sum[i]/NR, sqrt((sumsq[i]-sum[i]^2/NR)/NR)}
         }' dats/$version-$i.dat) 
    
    #saving the number of threads, avgs, and stdevs to generate graphs
    printf "%-20s %s\n" "$i" "$results" >> ../plot/$version.dat
    rm dats/$version-$i.dat
done
