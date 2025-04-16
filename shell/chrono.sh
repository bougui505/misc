#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Apr 15 14:34:38 2025

# set -e  # exit on error
# set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -e, --elapsed print elapsed time since the start of the script
EOF
}

ELAPSED=0  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -e|--elapsed) ELAPSED=1 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

function print_line () {
    echo "$deltat $line"
    if [[ $ELAPSED -eq 0 ]]; then 
        t0=$(date +%s%3N)
    fi
}

# Get start time in milliseconds
t0=$(date +%s%3N)
while sleep 0.001s; do
    t1=$(date +%s%3N)
    deltat=$((t1 - t0))
    deltat=$(printf "%02d:%02d:%02d.%03d" $((deltat/3600000)) $(( (deltat%3600000)/60000 )) $(( (deltat%60000)/1000 )) $(( deltat%1000 )))
    read -r -t 0.001 line
    ret=$?
    if [[ $ret -eq 1 ]]; then  # this is the end of the input
        exit 0
    elif [[ $ret -eq 0 ]]; then  # a line was read
        print_line
    else  # print the elapsed time
        echo -ne "$deltat\r"
    fi
done < "${1:-/dev/stdin}"
