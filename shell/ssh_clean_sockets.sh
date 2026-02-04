#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Feb  4 15:33:58 2026

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

# Change path to where your sockets are stored (e.g., ~/.ssh/sockets/ or /tmp/)
SOCKET_DIR="$HOME/.ssh/sockets"

for socket in "$SOCKET_DIR"/*; do
    # -O check verifies if a master process is actually listening
    if ! ssh -O check -S "$socket" 2>/dev/null; then
        echo "Cleaning up dead socket: $socket"
        rm -f "$socket"
    # else
    #     # Optional: Exit active but idle sockets
    #     # ssh -O exit -S "$socket"
    fi
done
