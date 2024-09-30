#!/bin/bash

# Executáveis CPU e GPU
exec_cpu_parallel="num_primos_paralelo_normal"
exec_cpu_concurrent="num_primos_fila_concorrente_normal"
exec_gpu_parallel="num_primos_paralelo_sequencial_cuda"
exec_gpu_concurrent="num_primos_fila_concorrente_cuda"

# Diretórios de logs
log_cpu_parallel="logs/cpu_parallel"
log_cpu_concurrent="logs/cpu_concurrent"
log_gpu_parallel="logs/gpu_parallel"
log_gpu_concurrent="logs/gpu_concurrent"

# Parâmetros
problemSize=10000000

# Função para executar e salvar a saída padrão no log
run_and_log() {
    local executable=$1
    local args=$2
    local logfile=$3

    echo "Running: $executable with args: $args"
    ./$executable $args >> $logfile  # Redirecionando apenas a saída padrão (sem time)
    echo "Done: $executable" >> $logfile
    echo "-----------------------------------" >> $logfile
}

# Executa CPU paralelo de 1 a 12 threads
echo "== Executando CPU Paralelo =="
for j in {1..12..1}; do
    echo "Running CPU Paralelo with $j threads!"
    run_and_log $exec_cpu_parallel "$problemSize $j" "$log_cpu_parallel-$j.log"
done

# Executa GPU paralelo de 128 em 128 threads até 896
echo "== Executando GPU Paralelo =="
for j in {1..896..128}; do
    echo "Running GPU Paralelo with $j threads!"
    run_and_log $exec_gpu_parallel "$problemSize $j" "$log_gpu_parallel-$j.log"
done

# Executa CPU concorrente de 1 a 12 threads
echo "== Executando CPU Concorrente =="
for j in {1..12..1}; do
    echo "Running CPU Concorrente with $j threads!"
    run_and_log $exec_cpu_concurrent "$problemSize $j" "$log_cpu_concurrent-$j.log"
done

# Executa GPU concorrente de 128 em 128 threads até 896
echo "== Executando GPU Concorrente =="
for j in {1..896..128}; do
    echo "Running GPU Concorrente with $j threads!"
    run_and_log $exec_gpu_concurrent "$problemSize $j" "$log_gpu_concurrent-$j.log"
done
