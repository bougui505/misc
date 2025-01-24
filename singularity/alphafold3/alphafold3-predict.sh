#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jan 24 13:42:07 2025
# See: https://docs.pasteur.fr/display/FAQA/How+to+run+Alphafold+3

#SBATCH -N 1
#SBATCH --cpus-per-task=8                  # nhmmer // jackhmmer default requirement
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH -J alphafold3

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection
set -x

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

module load apptainer

ALPHAFOLD3_DATA="/opt/gensoft/data/alphafold/3.0.0"
MODEL_DIR="/pasteur/appa/homes/bougui/alphafold3"
SIFFILE="/pasteur/appa/homes/bougui/alphafold3/alphafold3.sif"
cmd_args="--db_dir=${ALPHAFOLD3_DATA}\
          --model_dir=${MODEL_DIR}\
          $@"

apptainer run \
  -B /local/databases/rel/alphafold3 \
  -B $ALPHAFOLD3_DATA \
  --nv $SIFFILE \
  python3 /app/alphafold/alphafold3/run_alphafold.py ${cmd_args}
