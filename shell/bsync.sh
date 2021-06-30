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
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

function usage () {
    cat << EOF
bsync help
bsync -- bidirectional sync -- uses rsync to sync directories in both direction
    -h, --help print this help message and exit
    -d1, --dir1 source directory
    -d2, --dir2 secondary directory
    --delete delete files in dir2 not present in dir1
Usage:
    bsync -d1 dir1 -d2 dir2
This command will run 
    rsync dir1 dir2 
and then 
    rsync dir2 dir1
EOF
}

if [[ "$#" -eq 0 ]]; then
    usage
    exit 1
fi

DELETE=0
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -d1|--dir1) DIR1="$2"; shift ;;
        -d2|--dir2) DIR2="$2"; shift ;;
        --delete) DELETE=1 ;;
    esac
    shift
done

function _rsync_ () {
    if [[ $DELETE -eq 1 ]]; then
        rsync -a -zz --update --info=progress2 -h --delete --backup --backup-dir bkp/$(date +%Y%m%d_%H:%M:%S:%N) --exclude=".*" --exclude=".*/" --exclude bkp --exclude .git --exclude .history.dir.rec $1 $2
    else
        rsync -a -zz --update --info=progress2 -h --exclude=".*" --exclude=".*/" --exclude .git --exclude .history.dir.rec $1 $2
    fi
}

if [[ $DELETE -eq 1 ]]; then
    read "?--delete option. Files present in $DIR2 but not in $DIR1 will be moved to a bkp directory. Are you sure ? (Y-y/N-n) " REPLY
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        exit 1
    fi
fi

echo "Syncing: $DIR1 -> $DIR2"
_rsync_ $DIR1 $DIR2
echo "Syncing: $DIR1 <- $DIR2"
_rsync_ $DIR2 $DIR1
