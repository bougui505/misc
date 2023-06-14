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
Use TMalign to align protein pockets.
Returns the TMscore average for normalized on length of protein1 and length of protein2
    -s1, --struct1 First protein structure file
    -s2, --struct2 Second protein structure file
    -l1, --lig1 First selection or file for ligand1
    -l2, --lig2 First selection or file for ligand2
    -r, --radius Radius in Angstrom to define the pocket around the ligand
    --raw Display raw TMalign output
    -h, --help print this help message and exit
EOF
}

RAW=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -s1|--struct1) STRUCT1="$2"; shift ;;
        -s2|--struct2) STRUCT2="$2"; shift ;;
        -l1|--lig1) LIG1="$2"; shift ;;
        -l2|--lig2) LIG2="$2"; shift ;;
        -r|--radius) RADIUS="$2"; shift ;;
        --raw) RAW=1 ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

$DIRSCRIPT/pocket_selector.py -p $STRUCT1 -l $LIG1 -r $RADIUS -o $MYTMP/pocket1.pdb
$DIRSCRIPT/pocket_selector.py -p $STRUCT2 -l $LIG2 -r $RADIUS -o $MYTMP/pocket2.pdb
# Print the average TMscore normalized by lenght of protein1 and protein2
TMalign $MYTMP/pocket1.pdb $MYTMP/pocket2.pdb -outfmt 2 > $MYTMP/out.txt 
if [ $RAW -eq 1 ]; then
    cat $MYTMP/out.txt | column -t
else
    cat $MYTMP/out.txt | awk 'NR==2{print ($3+$4)/2}'
fi
