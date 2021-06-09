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
Get all the tmscores for the given system
    -m, --model get all the tmscores for the given model
    -n, --native get all the tmscores for the given native
    -i, --inp input recfile with tmscores as produced by tmscore-multi
    -o, --out optional output rec file to store the results
    --max only print the maximum value
    -h, --help print this help message and exit
EOF
}

MAX=0
MODEL="None"
NATIVE="None"
OUT="None"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--model) MODEL="$2"; shift ;;
        -n|--native) NATIVE="$2"; shift ;;
        --max) MAX=1 ;;
        -i|--inp) RECFILE="$2"; shift ;;
        -o|--out) OUT="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

# columns header: "model","native","tmscore"
TMSCORE=$(
if (test "$MODEL" != "None"); then
    rec2csv $RECFILE \
        | sed 's/^"//' | sed 's/"$//' | sed 's/","/,/g' \
        | awk -F"," -v"MODEL=$MODEL" '{if ($1==MODEL){print $3}}' \
        | ((test $MAX -eq 1) && awk 'BEGIN{MAX=0}{if ($1>MAX){MAX=$1}}END{print MAX}' || cat)
fi
if (test "$NATIVE" != "None"); then
    rec2csv $RECFILE \
        | sed 's/^"//' | sed 's/"$//' | sed 's/","/,/g' \
        | awk -F"," -v"NATIVE=$NATIVE" '{if ($2==NATIVE){print $3}}' \
        | ((test $MAX -eq 1) && awk 'BEGIN{MAX=0}{if ($1>MAX){MAX=$1}}END{print MAX}' || cat)
fi
)

if (test "$NATIVE" != "None"); then
    KEY='native'
    VALUE=$NATIVE
else
    KEY='model'
    VALUE=$MODEL
fi

if (test "$OUT" = "None") then
    echo $TMSCORE 
else
    touch $OUT
fi

if (test "$OUT" != "None"); then
    flock $OUT cat << EOF >> $OUT
${KEY}: $VALUE
tmscore: $TMSCORE

EOF
fi
