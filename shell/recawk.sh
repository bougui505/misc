#!/usr/bin/env bash
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

bold=$(tput bold)
normal=$(tput sgr0)

function usage () {
    cat << EOF

    -h, --help print this help message and exit

------------------------------------------------

Read a rec file formatted as:

${bold} 
key1=val1
key2=val2
--
key1=val12
key2=val22
--
[...]
${normal} 

using awk.

An example rec file can be found in ${bold}$DIRSCRIPT/recawk_test/data/file.rec.gz${normal}

The key, value couples are stored in ${bold}rec${normal} awk array.
To access key1, use:

    ${bold}rec["key1"]${normal} -> val1 (if in first records)

    ${bold}zcat data/file.rec.gz | recawk '{print rec["i"]}'${normal}

The full rec file is not stored in ${bold}rec${normal}. Just the current record is stored.

To enumerate fields just use:

    ${bold} 
    for (field in rec){
        print field
    }
    ${normal} 

A function ${bold}printrec()${normal} can be used to print the current record. The record separator "--" is not printed by ${bold}printrec()${normal} to allow the user to add an item to the record:

    ${bold}zcat data/file.rec.gz | recawk '{printrec();print("k=v");print("--")}'${normal}

Variable ${bold}nr${normal} is defined. ${bold}nr${normal} is the number of input records awk has processed since the beginning of the program’s execution. Not to be confused with ${bold}NR${normal}, which is the builtin awk variable, which store the number of lines awk has processed since the beginning of the program’s execution.

    ${bold}zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("NR="NR);print("--")}'${normal}

Examples:
    
${bold} 
    zcat data/file.rec.gz | recawk '{print rec["i"]}'
    zcat data/file.rec.gz | recawk '{for (field in rec){print field}}'
    zcat data/file.rec.gz | recawk '{printrec();print("k=v");print("--")}
    zcat data/file.rec.gz | recawk '{printrec();print("nr="nr);print("NR="NR);print("--")}'
${normal} 

EOF
}

case $1 in
    -h|--help) usage; exit 0 ;;
esac

if [ "$#" -eq 0 ]; then
    usage; exit 0
fi

CMD=$1
FILENAMES="${@:2}"

awk -F"=" '
function printrec(){
    for (field in rec){
        print field"="rec[field]
    }
}
BEGIN{
nr=0
}
{
if ($0=="--"){
    nr+=1
    '"$CMD"'
    delete rec
}
else{
    rec[$1]=substr($0,length($1)+2)
}
}' $FILENAMES
