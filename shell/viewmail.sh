#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Sep 23 12:34:42 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script
# 
# function usage () {
#     cat << EOF
# Help message
#     -h, --help print this help message and exit
# EOF
# }
# 
# N=1  # Default value
# while [ "$#" -gt 0 ]; do
#     case $1 in
#         -n|--number) N="$2"; shift ;;
#         -h|--help) usage; exit 0 ;;
#         --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
#         *) usage; exit 1 ;;
#     esac
#     shift
# done

# package: maildir-utils
mu view  $(realpath $1)
