#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Mar 19 14:14:46 2024
# Manage ext4 metadata

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Manage ext4 metadata
    -h, --help print this help message and exit
    -c, --comment add the given comment
    -f, --file file to add the comment on
    -l, --list list file along with comments
EOF
}

function listmeta () {
    # See: https://stackoverflow.com/a/68718493/1679629
    ls -1 | while read -r FILE; do
        comment=`xattr -p user.comment "$FILE" 2>/dev/null`
        if [ -n "$comment" ]; then
            echo "$FILE Comment: $comment"
        else
            echo "$FILE"
        fi
    done
}

while [ "$#" -gt 0 ]; do
    case $1 in
        -c|--comment) COMMENT="$2"; shift ;;
        -f|--file) FILE="$2"; shift ;;
        -l|--list) listmeta; exit 0 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

if [[ ! -z $COMMENT ]] && [[ ! -z $FILE ]]; then
    xattr -w user.comment "$COMMENT" $FILE
fi
