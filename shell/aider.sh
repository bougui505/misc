#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Dec 15 15:06:27 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

export OLLAMA_API_BASE=http://127.0.0.1:11435

# Check if port 11435 is already listening
if lsof -i :11435 > /dev/null 2>&1; then
    echo "SSH tunnel already established on port 11435"
    # Test if the tunnel is working by checking the OLLAMA API
    if curl -s --fail http://localhost:11435/api/tags > /dev/null; then
        echo "OLLAMA API is accessible through the tunnel"
    else
        echo "ERROR: OLLAMA API not accessible through the tunnel"
        exit 1
    fi
else
    echo "Establishing SSH tunnel..."
    ssh -f -N -T -L 11435:localhost:11435 dgx-spark
    
    # Wait a moment to ensure the SSH tunnel is established
    sleep 2
    
    # Test if the tunnel is working by checking the OLLAMA API
    if curl -s --fail http://localhost:11435/api/tags > /dev/null; then
        echo "OLLAMA API is accessible through the tunnel"
    else
        echo "ERROR: OLLAMA API not accessible through the tunnel"
        exit 1
    fi
fi

$DIRSCRIPT/aider_apptainer.sh --edit-format diff $@
