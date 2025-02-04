#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Feb  4 11:59:27 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -xo, --xoffset
    -yo, --yoffset
    -xs, --xscale
    -ys, --yscale
EOF
}

XOFFSET=0
YOFFSET=0
XSCALE=1
YSCALE=1
while [ "$#" -gt 0 ]; do
    case $1 in
        -xo|--xoffset) XOFFSET="$2"; shift ;;
        -yo|--yoffset) YOFFSET="$2"; shift ;;
        -xs|--xscale) XSCALE="$2"; shift ;;
        -ys|--yscale) YSCALE="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

cnee --record --mouse | awk  -v dirscript=$DIRSCRIPT -v xoffset=$XOFFSET -v yoffset=$YOFFSET -v xscale=$XSCALE -v yscale=$YSCALE '/7,4,0,0,1/ { system(dirscript"/mouseloc.sh -xo "xoffset" -yo "yoffset" -xs "xscale" -ys "yscale)}'
