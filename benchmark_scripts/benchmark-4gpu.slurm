#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --gres=gpu:4
#SBATCH --time=01:59:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=24
module load Keras/2.1.5-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
echo Running Benchmark with 4 GPUs
time python3 ~/benchmark_src/benchmark-GPU-4.py 500 100
