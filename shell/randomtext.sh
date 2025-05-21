#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed May 21 09:47:25 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Generate text with random words taken from /usr/share/dict/words

    -h, --help print this help message and exit
    -s, --size <int><unit> size of the text file to generate
        <int> is the size of the file
        <unit> is the unit of the size. It can be:
            K for kilobytes
            M for megabytes
            G for gigabytes
    -n, --number <int> number of words to generate
    -w, --width <int> width of the text file to generate (number of words per line)
    -l, --lines <int> number of lines to generate
        -w must be used with -l to generate a file with the specified number of lines
        with -w giving the number of words per line
EOF
}

# If no arguments are given, print the help message
if [ "$#" -eq 0 ]; then
    usage
    exit 0
fi

while [ "$#" -gt 0 ]; do
    case $1 in
        -s|--size) SIZE="$2"; shift ;;
        -n|--number) NUMBER="$2"; shift ;;
        -w|--width) WIDTH="$2"; shift ;;
        -l|--lines) LINES="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

# Check if LINES is set
if [ ! -z "$LINES" ]; then
    # Check if WIDTH is set
    if [ ! -z "$WIDTH" ]; then
        NWORDS=$((LINES * WIDTH))
        shuf -r /usr/share/dict/words | head -n $NWORDS \
            | awk -v"WIDTH=$WIDTH" '{printf $0" "; if (NR % WIDTH == 0) {print ""}}' \
            | head -n $LINES
        exit 0
    else
        echo "ERROR: -n must be used with -w to generate a file with the specified number of lines with -w giving the number of words per line" >&2
        exit 1
    fi
fi
# Check if SIZE is set
if [ ! -z "$SIZE" ]; then
    if [ ! -z "$WIDTH" ]; then
        shuf -r /usr/share/dict/words \
            | awk -v"WIDTH=$WIDTH" '{printf $0" "; if (NR % WIDTH == 0) {print ""}}' \
            | head -c "$SIZE"
        exit 0
    else
        shuf -r /usr/share/dict/words | tr '\n' ' ' | head -c "$SIZE"
        exit 0
    fi
fi
# Check if NUMBER is set
if [ ! -z "$NUMBER" ]; then
    shuf -r /usr/share/dict/words | head -n "$NUMBER" | tr '\n' ' '
    exit 0
fi
