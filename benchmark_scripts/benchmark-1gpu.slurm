#!/bin/bash
#SBATCH --partition gpgpu
#SBATCH --gres=gpu
#SBATCH --time=00:30:00
#SBATCH --mem=70G
#SBATCH --cpus-per-task=24
module load Keras/2.1.5-intel-2017.u2-GCC-6.2.0-CUDA9-Python-3.6.4-GPU
echo Running Benchmark with single GPU
time python3 ~/benchmark_src/benchmark-GPU.py
