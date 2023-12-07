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
for (i=1;i<=NF;i++){
    data[ARGIND][FNR][i]=$i
}
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
i=0
j=0
for (argind=1;argind<=nfiles;argind++){
    for (nf=1;nf<=nfmax;nf++){
        j++
        i=0
        for (nr=1;nr<=nrmax;nr++){
            i++
            if (data[argind][nr][nf]!=""){
                out[i][j] = data[argind][nr][nf]
            }
            else{
                out[i][j] = "-"
            }
        }
    }
    printf("\n")
}
for (i2=1;i2<=i;i2++){
    for (j2=1;j2<=j;j2++){
        printf out[i2][j2]" "
    }
    printf "\n"
}
}
' $@
