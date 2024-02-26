#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Feb 26 11:36:00 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
# set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

URL=$(cat $DIRSCRIPT/gcal_ical_url.txt)

curl "$URL" | singularity run $DIRSCRIPT/../singularity/bougui.sif ics2rem > $HOME/gcal_bougui.rem
