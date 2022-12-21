#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
EOF
}

N=1  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

# See: https://www.gnu.org/software/ddrescue/manual/ddrescue_manual.html#Optical-media
ddrescue -n -b2048 /dev/dvd dvdimage mapfile
ddrescue -d -r1 -b2048 /dev/dvd dvdimage mapfile
