#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Feb 11 14:19:34 2026

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Edit the given target in the current Makefile
    edit-target target
EOF
}

if [[ $# -eq 0 ]]; then
    usage; exit 1
fi

edit-target () {
	local target=$1 
	local line=$(grep -nE "^$target:" Makefile | cut -d: -f1 | head -n 1) 
	if [ -n "$line" ]
	then
		lvim Makefile +$line
	else
		echo "Target '$target' not found in Makefile."
	fi
}

edit-target $1
