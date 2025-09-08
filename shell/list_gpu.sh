#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Sep  8 09:50:22 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
List the GPU available on slurm on the given host.
    -h, --help print this help message and exit
    --host HOSTNAME specify the host to connect to (default: maestro)
EOF
}

HOST="maestro"  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        --host) HOST="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

ssh $HOST 'module load slurm;sinfo --format="%30N|%.22f|%.20G|%.10A|%.15E"'
echo ""
echo "To specify a gpu in slurm use:"
echo "#SBATCH --gres=gpu:<gpu-name>:1"
echo "for example:"
echo "#SBATCH --gres=gpu:A100:1"
echo ""
echo "GPU memory can be enforce using:"
echo "#SBATCH --gpu-memory=80G"
