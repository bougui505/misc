#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jan 23 08:39:06 2026

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection


DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

# Default port
PORT=11435
# Default host
HOST=dgx-spark

# Parse command line arguments
FLAG=1
while [[ $# -gt 0 && $FLAG -gt 0 ]]; do
    case $1 in
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        -H|--host)
            HOST="$2"
            shift 2
            ;;
        *)
            echo "Passing option $1 to aider"
            FLAG=0
            ;;
    esac
done

export OLLAMA_API_BASE=http://127.0.0.1:$PORT

# Check if port is already listening
if lsof -i :$PORT > /dev/null 2>&1; then
    echo "SSH tunnel already established on port $PORT"
    # Test if the tunnel is working by checking the OLLAMA API
    if curl --max-time 2 -s --fail http://localhost:$PORT/api/tags > /dev/null; then
        echo "OLLAMA API is accessible through the tunnel"
    else
        echo "ERROR: OLLAMA API not accessible through the tunnel"
        echo "Killing existing SSH tunnel and restarting..."
        # Kill the existing SSH tunnel
        lsof -ti :$PORT | xargs kill -9 2>/dev/null || true
        # Establish new SSH tunnel
        echo "Establishing SSH tunnel..."
        ssh -f -N -T -L $PORT:localhost:$PORT $HOST
        
        # Wait a moment to ensure the SSH tunnel is established
        sleep 2
        
        # Test if the tunnel is working by checking the OLLAMA API
        if curl --max-time 2 -s --fail http://localhost:$PORT/api/tags > /dev/null; then
            echo "OLLAMA API is accessible through the tunnel"
        else
            echo "ERROR: OLLAMA API not accessible through the tunnel after restart"
            exit 1
        fi
    fi
else
    echo "Establishing SSH tunnel..."
    ssh -f -N -T -L $PORT:localhost:$PORT $HOST
    
    # Wait a moment to ensure the SSH tunnel is established
    sleep 2
    
    # Test if the tunnel is working by checking the OLLAMA API
    if curl --max-time 2 -s --fail http://localhost:$PORT/api/tags > /dev/null; then
        echo "OLLAMA API is accessible through the tunnel"
    else
        echo "ERROR: OLLAMA API not accessible through the tunnel"
        exit 1
    fi
fi

export ANTHROPIC_AUTH_TOKEN=ollama
export ANTHROPIC_BASE_URL=http://localhost:$PORT

claude --model qwen3-coder
