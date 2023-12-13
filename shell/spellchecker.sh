#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Dec 13 08:56:44 2023

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

echo "Checking english ? [y,n]"
read answer
if [ $answer = "y" ]; then
  /usr/bin/ispell -d american $1
fi

echo "Checking french ? [y,n]"
read answer
if [ $answer = "y" ]; then
  /usr/bin/ispell -d french $1
fi
