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
    -t, --time change the default delay time (5 minutes) to the given number of minutes
    -a, --all also display hidden files
    -d, --day, number of day to put the delay (1 day = 1440 minutes)
EOF
}

DELAY=5  # delay in minutes
ALL=0
DAYS=1
while [ "$#" -gt 0 ]; do
    case $1 in
        -t|--time) DELAY="$2"; shift ;;
        -d|--day) DAYS="$2"; DELAY=$(qalc -t "$DAYS * 1440") ; shift ;;
        -a|--all) ALL=1 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

FILES=$(find -L . -type f -mmin -$DELAY | sed 's,./,,') 
if [ $ALL -eq 0 ]; then
    FILES=$(echo $FILES | tr " " "\n" | grep -v -e "^\." -e "/\." -e "__pycache__")
fi
if [[ ! -z $FILES ]]; then
    echo $FILES | xargs lt
fi
