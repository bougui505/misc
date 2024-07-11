#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jul 11 10:49:20 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

SMI1=$1
SMI2=$2

$DIRSCRIPT/_LSalign_smi_.sh $SMI1 $SMI2 \
    | recawk '{print rec["PC-score1"],rec["PC-score2"],rec["PC-score_max"],rec["Pval1"],rec["Pval2"],rec["jaccard"],rec["rmsd"],rec["size1"],rec["size2"]}'
