#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
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
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

function usage () {
    cat << EOF
Compute TMscore between multiple structures pairwisely given two list of pdb files
    -h, --help print this help message and exit
    --model, text file with path to model pdb
    --native, text file with path to native pdb
    --out, output file name (recfile)
    --njobs, number of parallel jobs (default: 100)
EOF
}

NJOBS=100
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --model) MODELS="$2"; shift ;;
        --native) NATIVES="$2"; shift ;;
        --out) OUT="$2"; shift ;;
        -n|--njobs) NJOBS="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

(test -z $MODELS) && (usage; exit 1)
(test -z $NATIVES) && (usage; exit 1)
(test -z $OUT) && (usage; exit 1)

(test -f $OUT) && echo "file exists: $OUT" && exit 1
touch $OUT

OUTDIR=$(date +%s)
mkdir $OUTDIR

tsp -C
tsp -S $NJOBS

i=1000000
PROGRESS=0
TOTAL=$(wc -l $NATIVES)
for NATIVE in $(sort -u $NATIVES); do
    (( PROGRESS += 1 ))
    for MODEL in $(sort -u $MODELS); do
        (( i+=1 ))
        tsp -n $DIRSCRIPT/tmscore_format.sh $MODEL $NATIVE $OUTDIR/$i.out
    done
    echo -ne "$PROGRESS/$TOTAL           \r"
    tsp -w  # Wait for the last job
    sleep .2
    find $OUTDIR -type f -exec cat {} + >> $OUT && rm -r $OUTDIR  && mkdir $OUTDIR  # Handle long list of files to concatenate
done
echo ""

rmdir $OUTDIR
