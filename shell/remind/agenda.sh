#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Feb 27 23:38:21 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

source $HOME/source/hhighlighter/h.sh

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
Print an agenda using remind: rem -b1 -n
with an optionnal start date (YYYY/MM/DD):
rem -b1 -n YYYY/MM/DD

Usage:
    -h, --help print this help message and exit
    -d, --date date to start from (YYYY/MM/DD)
    -f, --fzf use fzf to fuzzy search in the agenda
EOF
}

DATE=""
FZF=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -d|--date) DATE=$2; shift ;;
        -f|--fzf) FZF=1 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
    esac
    shift
done


function agenda () {
    if [[ -z $1 ]]; then
        DATE=$(date +%F | tr - /)
    else
        DATE=$1  # Optionnal start date (YYYY/MM/DD)
    fi
    rem -b1 -n $DATE | sort -r | h $DATE MALO MAUD GUIL VACS OFF_ GUIT PASTEUR
}

if [[ $FZF -eq 1 ]]; then
    agenda | tac | fzf -m --color --ansi
else
    agenda $DATE
fi
