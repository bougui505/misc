#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-03-17

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Unique command using awk
    -h, --help print this help message and exit
    -r, --refresh_rate refresh rate for printing the count (default 100, print every 100 lines)
EOF
}

REFRESH_RATE=100
while [ "$#" -gt 0 ]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -r|--refresh_rate) REFRESH_RATE=$2; shift ;;
        *) usage; exit 1 ;;
    esac
    shift
done

awk -v refresh_rate=$REFRESH_RATE '
function printcount () {
# \033[H moves cursor to top, \033[J clears the screen
  printf "\033[H\033[J"
  for (k in COUNT){
    print k,COUNT[k]
  }
}
{
COUNT[$0]++
if (NR % refresh_rate == 0) {
  printcount()
}
}
END{
  printcount()
}'
