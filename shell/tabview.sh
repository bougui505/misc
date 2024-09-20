#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Sep 19 13:02:07 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
View a table in vim
    -h, --help print this help message and exit
EOF
}

N=1  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done


awk '{if (NR==1){
        for (i=1;i<=NF;i++){
            printf $i"["i"] "
        }
        print ""
    }
    else{
        print $0
    }
}' \
    | column -t \
    | view -c ':1split' -c ':wincmd j' -c ':set cursorline' -c ':set cursorcolumn' -c ':colorscheme desert' -
