#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Feb  8 10:50:03 2024
#
# Use bard to write a git commit message from a git diff

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
Use bard to write a git commit message from a git diff
    -c, --context add context in the git diff
    -a, --add write a git commit message from the given new file (git add)
    -h, --help print this help message and exit
EOF
}

CONTEXT=0
ADD=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -c|--context) CONTEXT=1 ;;
        -a|--add) ADD="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

if [[ "$ADD" != "0" ]]; then
    # (cat $ADD) | bard 'write a git commit message from the given diff'
    echo "Write a git commit message for adding the given content below of a file named $ADD:"
    cat $ADD
    exit 0
fi
if [[ $CONTEXT -eq 0 ]]; then
    echo "Write a git commit from the given diff:"
    git diff --unified=0
else
    echo "Write a git commit from the given diff:"
    git diff
fi
