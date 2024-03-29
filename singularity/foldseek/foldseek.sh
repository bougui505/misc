#!/usr/bin/env zsh
# shellcheck shell=bash
# -*- coding: UTF8 -*-

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rvf "$MYTMP"' EXIT KILL INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Help message
    -p, --pdb pdbfile to search for
EOF
}

while [ "$#" -gt 0 ]; do
    case $1 in
        -p|--pdb) PDB="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

INPDB=$MYTMP/$PDB:t
echo $INPDB
if [ ! -f $PDB ]; then
    echo "Downloading pdb $PDB"
    getpdb -p $PDB --out $INPDB.pdb
else
    cp -v $PDB $INPDB.pdb
fi

# DB can be /opt/pdb or /opt/afdb
singularity exec foldseek.sif foldseek easy-search $INPDB.pdb /opt/afdb ${PDB:r}_foldseek.out $MYTMP
