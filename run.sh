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
problemSize=500000000

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
for j in {1..12}; do
    echo "Running CPU Paralelo with $j threads!"
    run_and_log $exec_cpu_parallel "$problemSize $j" "$log_cpu_parallel-$j.log"
done

# Executa GPU paralelo com múltiplos de 64 threads
echo "== Executando GPU Paralelo =="
gpu_threads=(64 128 192 256 320 384 448 512 576 640 704 768 832)  # Múltiplos de 64
for j in "${gpu_threads[@]}"; do
    echo "Running GPU Paralelo with $j threads!"
    run_and_log $exec_gpu_parallel "$problemSize $j" "$log_gpu_parallel-$j.log"
done

# Executa CPU concorrente de 1 a 12 threads
echo "== Executando CPU Concorrente =="
for j in {1..12}; do
    echo "Running CPU Concorrente with $j threads!"
    run_and_log $exec_cpu_concurrent "$problemSize $j" "$log_cpu_concurrent-$j.log"
done

# Executa GPU concorrente com múltiplos de 64 threads
echo "== Executando GPU Concorrente =="
for j in "${gpu_threads[@]}"; do
    echo "Running GPU Concorrente with $j threads!"
    run_and_log $exec_gpu_concurrent "$problemSize $j" "$log_gpu_concurrent-$j.log"
done
