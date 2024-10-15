#!/bin/bash

# Criar a pasta 'dats' se ela não existir
if [ ! -d "dats" ]; then
  mkdir dats  # Cria o diretório 'dats' para armazenar os dados
fi

# Array que contém os tipos de execução (CPU e GPU)
versions=("cpu_parallel" "cpu_concurrent" "gpu_parallel" "gpu_concurrent")

# Limpar os arquivos .dat antigos na pasta ../plot/ antes de gerar novos
for version in "${versions[@]}"; do
  rm -f ../plot/$version.dat  # Remove qualquer arquivo .dat existente na pasta ../plot
done

# Lista de valores de threads para a GPU em múltiplos de 64 (começando de 64)
gpu_threads=(64 128 192 256 320 384 448 512 576 640 704 768 832)

# Coletar dados para CPU (de 1 a 12 threads) e GPU (com valores de gpu_threads)
for version in "${versions[@]}"; do
  if [[ "$version" == *gpu* ]]; then
    # GPU: Usar a lista de gpu_threads
    for i in "${gpu_threads[@]}"; do
      log_file="$version-$i.log"  # Nome do arquivo de log da GPU
      if [ -f "$log_file" ]; then
        # Coletar o tempo de execução do arquivo de log (padrão: "Tempo de execução: X segundos")
        exec_time=$(awk '/Tempo de execução/ { print $5 }' $log_file)

        # Salvar o número de threads e o tempo de execução no arquivo .dat correspondente
        echo "$i $exec_time" >> ../plot/$version.dat
      else
        # Se o arquivo de log não for encontrado
        echo "Arquivo de log $log_file não encontrado!"
      fi
    done
  else
    # Se for uma versão de CPU, itera de 1 a 12 threads
    for ((i=1; i<=12; i++)); do
      log_file="$version-$i.log"  # Nome do arquivo de log da CPU
      if [ -f "$log_file" ]; then
        # Coletar o tempo de execução do arquivo de log
        exec_time=$(awk '/Tempo de execução/ { print $5 }' $log_file)

        # Salvar o número de threads e o tempo de execução no arquivo .dat correspondente
        echo "$i $exec_time" >> ../plot/$version.dat
      else
        # Se o arquivo de log não for encontrado
        echo "Arquivo de log $log_file não encontrado!"
      fi
    done
  fi
done
