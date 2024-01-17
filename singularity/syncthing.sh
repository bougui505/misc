#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Jan 17 10:09:24 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

# singularity run -B $(pwd) -B /c7/scratch2 --nv $DIRSCRIPT/bougui.sif $@

CMD="singularity run -B $(pwd)"

[ -e /c7/scratch2 ] &&  CMD+=" -B /c7/scratch2"
CMD+=" $DIRSCRIPT/bougui.sif syncthing --home ~/syncthing"
echo $CMD
$CMD
