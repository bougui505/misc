#!/usr/bin/env zsh
# shellcheck shell=bash
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #    
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT KILL INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Join files given in column based on the given columns
    -h, --help print this help message and exit
    -c, --col column number (starting from 1) to base the joining (e.g. 1)
EOF
}

TEST=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -c|--col) COL="$2"; shift ;;
        --test) TEST=1;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

if [ $TEST -eq 1 ];then
    OTHER=(data/a.txt data/b2.txt data/c2.txt)
    COL=1
    echo "# Testing with --col $COL -- $OTHER"
fi

awk -v "col=$COL" '
BEGIN{
    maxnf=0
    maxfnr=0
}
{
    i+=1
    x[$col]=$col
    if (FNR>maxfnr){
        maxfnr=FNR
    }
    if (FNR==1){
        fnum+=1
    }
    for (j=1;j<=NF;j++){
        if (j!=col){
            if (j>maxnf){
                maxnf=j
            }
            data[fnum"_"$col"_"j]=$j
        }
    }
}
END{
    total_fnum = fnum
    n=i
    p=maxnf
    # print total_fnum, n, p
    for (i in x){
        for (fnum=1;fnum<=total_fnum;fnum++){
            for (j=1;j<=p;j++){
                if (j!=col){
                    key=fnum"_"x[i]"_"j
                    printf data[key]","
                }
            }
        }
        printf "\n"
    }
}    
' $OTHER
