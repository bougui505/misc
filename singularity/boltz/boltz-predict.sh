#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Dec  9 10:56:33 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

singularity run --nv $DIRSCRIPT/boltz.sif boltz predict $@ --cache /opt/boltz_data
