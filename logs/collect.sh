#!/bin/bash

# Criar a pasta dats se não existir
if [ ! -d "dats" ]; then
  mkdir dats
fi

# Array de versões (logs para CPU e GPU)
versions=("cpu_parallel" "cpu_concurrent" "gpu_parallel" "gpu_concurrent")

# Limpar os arquivos .dat antigos na pasta ../plot/
for version in "${versions[@]}"; do
  rm -f ../plot/$version.dat
done

# Lista de threads da GPU (incluindo 1 thread)
gpu_threads=(1 128 192 256 320 384 448 512 576 640 704 768 832)

# Coletar dados para CPU (de 1 a 12 threads) e GPU (com valores de gpu_threads)
for version in "${versions[@]}"; do
  if [[ "$version" == *gpu* ]]; then
    # GPU: Usar a lista de gpu_threads
    for i in "${gpu_threads[@]}"; do
      log_file="$version-$i.log"
      if [ -f "$log_file" ]; then
        # Coletar tempos de execução dos logs
        exec_time=$(awk '/Tempo de execução/ { print $5 }' $log_file)

        # Salvar número de threads e tempo de execução no arquivo .dat
        echo "$i $exec_time" >> ../plot/$version.dat
      else
        echo "Arquivo de log $log_file não encontrado!"
      fi
    done
  else
    # CPU: Iterar de 1 a 12 threads
    for ((i=1; i<=12; i++)); do
      log_file="$version-$i.log"
      if [ -f "$log_file" ]; then
        # Coletar tempos de execução dos logs
        exec_time=$(awk '/Tempo de execução/ { print $5 }' $log_file)

        # Salvar número de threads e tempo de execução no arquivo .dat
        echo "$i $exec_time" >> ../plot/$version.dat
      else
        echo "Arquivo de log $log_file não encontrado!"
      fi
    done
  fi
done
