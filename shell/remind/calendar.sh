#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Feb 27 23:52:22 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
Print a calendar using remind: rem -b1 -m -c+
with an optionnal start date (YYYY/MM/DD):
rem -b1 -m -c+ \$DATE
    -h, --help print this help message and exit
    -d, --date date to start from (YYYY/MM/DD)
    -w, --week number of weeks to display (default: 1)
EOF
}

DATE=""  # Default value
WEEK=1
while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--date) DATE="$2"; shift ;;
        -w|--week) WEEK=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

rem -b1 -m -c+$WEEK $DATE

