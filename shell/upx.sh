#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Mar 12 10:09:03 2024
#
# Run up (https://github.com/akavel/up) and copy the 
# up content to xclip

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

up --output-script $MYTMP/up.sh
CMD=$(cat $MYTMP/up.sh | grep -v "^#" | tr -d "\n")
echo $CMD
echo $CMD | tr -d "\n" | xclip
