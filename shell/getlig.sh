#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jun  6 14:16:24 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Get the pdb of the ligand in mol2 format
    -h, --help print this help message and exit
    --pdb pdb code
    --out out mol2 file
    --chain
    --resid
EOF
}

while [ "$#" -gt 0 ]; do
    case $1 in
        --pdb) PDB="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        --chain) CHAIN="$2"; shift ;;
        --resid) RESID="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

curl "https://models.rcsb.org/v1/$PDB/ligand?auth_asym_id=$CHAIN&auth_seq_id=$RESID&encoding=mol2" >! $OUT
