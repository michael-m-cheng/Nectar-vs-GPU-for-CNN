# Nectar-vs-GPU-for-CNN

This repository contains the source code used to create the Food recognition model using Keras to train a convolutional deep neural network. These python source file are also used to test the performance of a deep learning on Nectar Cloud compute instance versus Spartan HPC GPU cluster. Included is also Slurm scripts used to run jobs on Spartan HPC.
## Requirements
 - Python 3.6
 - Keras 2
 - Tensorflow(-gpu) >1.9.0

The deep learning task utilises food images sourced from https://www.vision.ee.ethz.ch/datasets_extra/food-101/, and the image will need to split into test and training sets before the code can be executed.

## Usage

1. Extract the `food-101/images` contents into the benchmark_src directory
2. Run `python3 split_data.py` to setup training and testing image directories
3. Run `python3 <benchmark source>.py <num training samples> <num test samples>`

*NOTE The benchmark source files have some parameters that need to be changed to be run on your local machine
For example:*
  - Data path (currently points to the Spartan project data directory)
  - Number of gpus
  - Number of processes to run for preprocessing
  - Training batch size

## Acknowledgements
This research was supported by use of the Nectar Research Cloud, a collaborative Australian research platform supported by the National Collaborative Research Infrastructure Strategy (NCRIS).

This research was undertaken using the LIEF HPC-GPGPU Facility hosted at the University of Melbourne. This Facility was established with the assistance of LIEF Grant LE170100200.
