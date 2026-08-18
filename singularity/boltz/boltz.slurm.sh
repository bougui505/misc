#!/usr/bin/env bash

#SBATCH -p gpu       # Partition name (gpu)
#SBATCH --qos=gpu  # Quality of service (gpu)
#SBATCH --gres=gpu:1       # Number of GPUs (1)
#SBATCH --mem-per-gpu=48G  # Request at least 48GB of GPU memory
#SBATCH --cpus-per-task=8  # Number of CPU cores (8)
# #SBATCH --mem=<memory>       # Memory (16GB)
#SBATCH -N 1       # Number of nodes (1)

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jan 22 11:38:26 2026

set -e  # exit on error

YAML=$1
OUTDIR=${2:-outputs/pull_down}
mkdir -p $OUTDIR
module load apptainer
mkdir -p /tmp/bougui
apptainer exec --env NUMBA_CACHE_DIR=/tmp/bougui/.numba_cache \
               --env TRITON_CACHE_DIR=/tmp/bougui/triton_cache \
               --env TORCH_HOME=/tmp/bougui/torch_cache \
               -B /pasteur/appa/homes \
               --nv boltz.sif boltz predict $YAML --use_msa_server --override --out_dir $OUTDIR
