#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Sep 30 10:23:09 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
# set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Open the NOTES.md file as an html file in google-chrome.

Usage:
    notes.sh -d
    notes.sh -h | --help

Options:
    -h, --help    Print this help message and exit.
    -d, --display Open the NOTES.md file as an html file in google-chrome.
EOF
}

function display () {
    pandoc -s -f markdown -t html --css $DIRSCRIPT/notes.css $DIRSCRIPT/NOTES.md > $DIRSCRIPT/notes.html
    google-chrome $DIRSCRIPT/notes.html
}

function edit () {
    vim $DIRSCRIPT/NOTES.md
}

# N=1  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        # -n|--number) N="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        -d|--display) display; exit 0 ;;
        -e|--edit) edit; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

