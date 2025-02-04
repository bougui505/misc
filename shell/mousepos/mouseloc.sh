#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Feb  4 12:19:36 2025
#
function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -xo, --xoffset
    -yo, --yoffset
    -xs, --xscale
    -ys, --yscale
EOF
}

XOFFSET=0
YOFFSET=0
XSCALE=1
YSCALE=1
while [ "$#" -gt 0 ]; do
    case $1 in
        -xo|--xoffset) XOFFSET="$2"; shift ;;
        -yo|--yoffset) YOFFSET="$2"; shift ;;
        -xs|--xscale) XSCALE="$2"; shift ;;
        -ys|--yscale) YSCALE="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

xdotool getmouselocation | awk -v xoffset=$XOFFSET -v yoffset=$YOFFSET -v xscale=$XSCALE -v yscale=$YSCALE '{split($1,x,":");split($2,y,":");print "("(x[2]-xoffset)/xscale","(y[2]-yoffset)/yscale")"}'
