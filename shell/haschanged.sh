#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Feb  8 09:52:40 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

DATABASE="$HOME/.md5.db.gz"

function usage () {
    cat << EOF
Check if a file has changed.
MD5 checksum are stored in $DATABASE
    -h, --help print this help message and exit
    -f, --file file to check
    -c, --clean clean the database. Reset all change checks.
    -d, --db specify a custom database file instead of using $DATABASE
EOF
}

N=1  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -c|--clean) /bin/rm -v $DATABASE; exit 0 ;;
        -d|--db) DATABASE="$2"; shift ;;
        -f|--file)  FILE="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

touch $DATABASE
MD51=$(md5sum $DATABASE)
REALPATH=$(realpath $FILE)
MD5=$(md5sum $REALPATH)
echo $MD5 | gzip >> $DATABASE
zcat $DATABASE | sort -u | gzip | sponge $DATABASE
MD52=$(md5sum $DATABASE)
if [[ "$MD51" == "$MD52" ]]; then
    echo 0
else
    echo 1
fi
