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
    seq 1 \
    | awk -v SMI1="$SMI1" -v SMI2="$SMI2" '{
        print "smi1="SMI1
        print "smi2="SMI2
        print "PC-score1=-1"
        print "PC-score2=-1"
        print "PC-score_max=-1"
        print "Pval1=-1"
        print "Pval2=-1"
        print "jaccard=-1"
        print "rmsd=-1"
        print "size1=-1"
        print "size2=-1"
        print "--"
        }'
    exit 0
}

echo $SMI1 | obabel -ismi --gen3d -o mol2 2> /dev/null | sed 's/\*\*\*\*\*/smi1/' > $MYTMP/smi1.mol2
echo $SMI2 | obabel -ismi --gen3d -o mol2 2> /dev/null | sed 's/\*\*\*\*\*/smi2/' > $MYTMP/smi2.mol2
($DIRSCRIPT/LSalign $MYTMP/smi1.mol2 $MYTMP/smi2.mol2 -rf 1 || echo "smi1 smi2 -1 -1 -1 -1 -1 -1 -1 -1") > $MYTMP/out.txt
# cat $MYTMP/out.txt
awk -v SMI1="$SMI1" -v SMI2="$SMI2" '/smi1/{
    s1=$3;s2=$4;pval1=$5;pval2=$6;j=$7;rmsd=$8;size1=$9;size2=$10
    if (s1>s2){smax=s1}else{smax=s2}
    print "smi1="SMI1
    print "smi2="SMI2
    print "PC-score1="s1
    print "PC-score2="s2
    print "PC-score_max="smax
    print "Pval1="pval1
    print "Pval2="pval2
    print "jaccard="j
    print "rmsd="rmsd
    print "size1="size1
    print "size2="size2
    print "--"
    exit
    }' $MYTMP/out.txt
