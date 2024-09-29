#!/bin/bash
exec1="num_primos_paralelo_normal"
#exec2="num_primos_fila_concorrente_normal"
saveLog1="logs/$exec1"
saveLog2="logs/$exec2"
problemSize=1000000
for ((i=1; i<=1; i++)) #10 repetitions
do
    for j in {1..12..1} #from 2 to 12 threads
    do
        echo "Running rep $i with $j threads!"
        ./$exec1 $problemSize $j &>> $saveLog1-$j.log
        ./$exec2 $problemSize $j &>> $saveLog2-$j.log
        
    done
done