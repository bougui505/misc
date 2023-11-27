#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Nov 27 13:26:30 2023

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
List all the files modified the last 5 minutes (using find)
    -h, --help print this help message and exit
    -d, --delay change the default delay (5 minutes) to the given number of minutes
EOF
}

DELAY=5  # delay in minutes
while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--delay) DELAY="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

find . -type f -mmin -$DELAY | xargs lt
