#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Mar 26 18:22:47 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

export OLLAMA_API_BASE=http://127.0.0.1:11435

# Check if port 11435 is already listening
if lsof -i :11435 > /dev/null 2>&1; then
    echo "SSH tunnel already established on port 11435"
else
    echo "Establishing SSH tunnel..."
    ssh -f -N -T -L 11435:localhost:11435 dgx-spark
    
    # Wait a moment to ensure the SSH tunnel is established
    sleep 2
fi

apptainer run --nv $DIRSCRIPT/shell_gpt.sif $DIRSCRIPT/_sgpt_.sh "$@"
