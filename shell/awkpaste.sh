#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Dec  6 10:26:18 2023

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
awkpaste file1 file2 [...]
    -h, --help print this help message and exit
EOF
}

case $1 in
    -h|--help) usage; exit 0 ;;
esac

awk '
{
data[ARGIND][FNR][NF]=$NF
nlines[ARGIND] += 1
if (ARGIND > nfiles){
    nfiles=ARGIND
}
if (NF > nfmax){
    nfmax=NF
}
if (FNR > nrmax){
    nrmax=FNR
}
}
END{
for (k in data){
    # print k
}
for (nr=1;nr<=nrmax;nr++){
    for (argind=1;argind<=nfiles;argind++){
        for (nf=1;nf<=nfmax;nf++){
            if (data[argind][nr][nf]!=""){
                printf data[argind][nr][nf]" "
            }
            else{
            printf "- "
            }
        }
    }
    printf("\n")
}
}
' $@
