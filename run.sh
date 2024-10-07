#!/bin/bash

# Definição dos executáveis para CPU e GPU
exec_cpu_parallel="num_primos_paralelo_normal"  # Executável para CPU em modo paralelo
exec_cpu_concurrent="num_primos_fila_concorrente_normal"  # Executável para CPU em modo concorrente
exec_gpu_parallel="num_primos_paralelo_sequencial_cuda"  # Executável para GPU em modo paralelo
exec_gpu_concurrent="num_primos_fila_concorrente_cuda"  # Executável para GPU em modo concorrente

# Definição dos diretórios de logs
log_cpu_parallel="logs/cpu_parallel"  # Caminho para os logs do CPU em paralelo
log_cpu_concurrent="logs/cpu_concurrent"  # Caminho para os logs do CPU em modo concorrente
log_gpu_parallel="logs/gpu_parallel"  # Caminho para os logs do GPU em paralelo
log_gpu_concurrent="logs/gpu_concurrent"  # Caminho para os logs do GPU em modo concorrente

# Parâmetros do problema
problemSize=500000000  # Tamanho do intervalo de números a ser processado (número total de números primos a serem verificados)

# Função para executar um programa e salvar a saída no arquivo de log
run_and_log() {
    local executable=$1  # Nome do executável a ser rodado
    local args=$2  # Argumentos a serem passados para o executável (tamanho do problema e número de threads)
    local logfile=$3  # Arquivo de log onde a saída será salva

    # Exibe uma mensagem indicando qual executável está rodando e com quais argumentos
    echo "Running: $executable with args: $args"
    # Executa o programa com os argumentos e redireciona a saída padrão para o arquivo de log
    ./$executable $args >> $logfile
    # Mensagem indicando que o executável foi concluído e adiciona ao log
    echo "Done: $executable" >> $logfile
    echo "-----------------------------------" >> $logfile
}

# Executa o programa para CPU em modo paralelo, variando de 1 a 12 threads
echo "== Executando CPU Paralelo =="
for j in {1..12}; do
    echo "Running CPU Paralelo with $j threads!"
    run_and_log $exec_cpu_parallel "$problemSize $j" "$log_cpu_parallel-$j.log"  # Chama a função com o executável, número de threads e log
done

# Executa o programa para GPU em modo paralelo, variando o número de threads (começando com 1 e depois valores progressivos)
echo "== Executando GPU Paralelo =="
gpu_threads=(1 128 192 256 320 384 448 512 576 640 704 768 832)  # Definição dos valores de threads para a GPU
for j in "${gpu_threads[@]}"; do
    echo "Running GPU Paralelo with $j threads!"
    run_and_log $exec_gpu_parallel "$problemSize $j" "$log_gpu_parallel-$j.log"  # Chama a função com o executável, número de threads e log
done

# Executa o programa para CPU em modo concorrente, variando de 1 a 12 threads
echo "== Executando CPU Concorrente =="
for j in {1..12}; do
    echo "Running CPU Concorrente with $j threads!"
    run_and_log $exec_cpu_concurrent "$problemSize $j" "$log_cpu_concurrent-$j.log"  # Chama a função com o executável, número de threads e log
done

# Executa o programa para GPU em modo concorrente, variando o número de threads (começando com 1 e depois valores progressivos)
echo "== Executando GPU Concorrente =="
for j in "${gpu_threads[@]}"; do
    echo "Running GPU Concorrente with $j threads!"
    run_and_log $exec_gpu_concurrent "$problemSize $j" "$log_gpu_concurrent-$j.log"  # Chama a função com o executável, número de threads e log
done
