#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Feb  4 11:59:27 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'format_calibfile; ([[ -f $ERRFILE ]] && cat $ERRFILE >&2); ([[ -f $STDFILE ]] && cat $STDFILE >&2); rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -xo, --xoffset
    -yo, --yoffset
    -xs, --xscale
    -ys, --yscale
    -c, --calibrate calibrate the (x, y) coordinates. Click on point (0, 0); (1, 0) and (0, 1)
EOF
}

XOFFSET=0
YOFFSET=0
XSCALE=1
YSCALE=1
CALIB=0
STDFILE=$MYTMP/mousepos.std
ERRFILE=$MYTMP/mousepos.err
CALIBFILE=.mousepos.calib
while [ "$#" -gt 0 ]; do
    case $1 in
        -xo|--xoffset) XOFFSET="$2"; shift ;;
        -yo|--yoffset) YOFFSET="$2"; shift ;;
        -xs|--xscale) XSCALE="$2"; shift ;;
        -ys|--yscale) YSCALE="$2"; shift ;;
        -c|--calibrate) CALIB=1 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done


function format_calibfile(){
    if [[ $CALIB -eq 1 ]]; then
        cat $CALIBFILE \
            | tr -d "()" \
            | tr "," " " \
            | sponge $CALIBFILE
    fi
}

if [[ $CALIB -eq 1 ]]; then
    echo "Calibration..."
    echo "Click on point (0, 0); (1, 0) and (0, 1)"
    STDFILE="$CALIBFILE"
fi
if [[ -f "$CALIBFILE" ]] && [[ $CALIB -eq 0 ]]; then
    echo "Reading calibration file: $CALIBFILE"
    cat $CALIBFILE
    XOFFSET=$(cat $CALIBFILE | awk '{X[NR]=$1}END{print X[1]}')
    YOFFSET=$(cat $CALIBFILE | awk '{Y[NR]=$2}END{print Y[1]}')
    XSCALE=$(cat $CALIBFILE | awk '{X[NR]=$1}END{print X[2]-X[1]}')
    YSCALE=$(cat $CALIBFILE | awk '{Y[NR]=$2}END{print Y[3]-Y[1]}')
    echo "XOFFSET="$XOFFSET
    echo "YOFFSET="$YOFFSET
    echo "XSCALE="$XSCALE
    echo "YSCALE="$YSCALE
    # exit 0
fi

(cnee --record --mouse | awk  -v dirscript=$DIRSCRIPT -v xoffset=$XOFFSET -v yoffset=$YOFFSET -v xscale=$XSCALE -v yscale=$YSCALE '/7,4,0,0,1/ { system(dirscript"/mouseloc.sh -xo "xoffset" -yo "yoffset" -xs "xscale" -ys "yscale)}') 2> $ERRFILE | tee $STDFILE
