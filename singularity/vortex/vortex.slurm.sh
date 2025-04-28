#!/usr/bin/env bash

#SBATCH -J vortex         # Job name
#SBATCH -o slurm/vortex_%j.out     # Output file name
#SBATCH -e slurm/vortex_%j.err     # Error file name
#SBATCH -p gpu       # Partition name (gpu)
#SBATCH --qos=gpu  # Quality of service (gpu)
#SBATCH --gres=gpu:l40s:1       # Number of GPUs (1)
#SBATCH --cpus-per-task=8  # Number of CPU cores (8)
#SBATCH -N 1      # Number of nodes (1)

#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Apr 28 13:57:20 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

module load apptainer
apptainer build --fakeroot vortex.sif vortex.def
