#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Jul 10 16:46:56 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

SMI1=$1
SMI2=$2

exit1 (){
    echo -1
    exit 1
}

# echo "SMI1: $SMI1"
# echo "SMI2: $SMI2"
RDKITFIX="$DIRSCRIPT/../../python/mols/rdkit_fix.py"
DOCKPREP="$DIRSCRIPT/../../python/chimera/dockprep.sh"
$RDKITFIX -s $SMI1 -o $MYTMP/smi1.sdf > /dev/null 2>&1 || exit1
$RDKITFIX -s $SMI2 -o $MYTMP/smi2.sdf > /dev/null 2>&1 || exit1
OUT1=$(mktemp -p . --suffix .mol2)
OUT2=$(mktemp -p . --suffix .mol2)
$DOCKPREP -i $MYTMP/smi1.sdf -o $OUT1 > /dev/null 2>&1 || exit1
$DOCKPREP -i $MYTMP/smi2.sdf -o $OUT2 > /dev/null 2>&1 || exit1
mv $OUT1 $MYTMP/smi1.mol2
mv $OUT2 $MYTMP/smi2.mol2
($DIRSCRIPT/LSalign $MYTMP/smi1.mol2 $MYTMP/smi2.mol2 -rf 1 || echo "done") > $MYTMP/out.txt
awk '/smi1.sdf/{
    s1=$3;s2=$4
    if (s1>s2){print s1}else{print s2}
    }' $MYTMP/out.txt
