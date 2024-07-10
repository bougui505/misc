#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Jul 10 11:25:40 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

IMGFILE=$1
OUTFILE=$IMGFILE:r.pdf
echo "$IMGFILE -> $OUTFILE"
inkscape $1 -o $OUTFILE
gs -sDEVICE=pdfwrite -dCompatibilityLevel=1.5 -dEmbedAllFonts=true -dPDFSETTINGS=/screen -dNOPAUSE -dQUIET -dBATCH -sOutputFile=$MYTMP/$OUTFILE:t $OUTFILE
mv $MYTMP/$OUTFILE:t $OUTFILE
