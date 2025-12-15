#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Mar 26 18:23:51 2025

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

# ollama pull deepseek-coder-v2
sgpt --model ollama/qwen3-coder:latest "$@"
