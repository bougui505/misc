#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jun 14 09:54:57 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

DIR=""
RES=102
WIDTH=1024
HEIGHT=768

function usage () {
    cat << EOF
Resample all png images in a directory to a given resolution
See: $DIRSCRIPT/Image_resolutions.pdf for advices
    -h, --help print this help message and exit
    -d, --dir directory with png images to resample
    -r, --res target resolution in DPI (default $RES DPI)
    -W, --width target width in pixels (default $WIDTH pixels)
    -H, --height target height in pixels (default $HEIGHT pixels)
EOF
}

while [ "$#" -gt 0 ]; do
    case $1 in
        -d|--dir) DIR="$2"; shift ;;
        -r|--res) RES=$2; shift ;;
        -W|--width) WIDTH=$2; shift ;;
        -H|--height) HEIGHT=$2; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

DIR=$(echo $DIR | sed 's,/\s*$,,')
OUTDIR="${DIR}_${RES}"
[[ ! -d $OUTDIR ]] && mkdir -v $OUTDIR

for PNG in $(ls $DIR/*.png); do
    OUTPNG="$OUTDIR/${PNG:t}"
    if [[ ! -f $OUTPNG ]]; then
        echo "$PNG -> $OUTPNG"
        convert -strip -resize ${WIDTH}x${HEIGHT}\> -resample $RES $PNG $OUTPNG
    fi
done
