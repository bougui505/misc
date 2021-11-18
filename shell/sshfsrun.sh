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

set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
DIRMNT="/mnt/sshfs"

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -m, --mount mount the directory given by --dir in $DIRMNT
    -u, --unmount unmount $DIRMNT
    -H, --host host to run the command on (see -r, default: desk)
    -r, --run command to run on host (see -H)
    -d, --dir mount the given directory in /mnt/sshfs (default desk://)
EOF
}

N=1  # Default value
MOUNT=0
UNMOUNT=0
HOST="desk"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        -m|--mount) MOUNT=1;;
        -u|--unmount) UNMOUNT=1;;
        -H|--host) HOST="$2"; shift ;;
        -d|--dir) DIR="$2"; shift ;;
        -r|--run) CMD="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        *) usage; exit 1 ;;
    esac
    shift
done

func checkmnt(){
    ISMNT=$(mount | grep $DIRMNT | wc -l)
    echo $ISMNT
}

func checkinmnt() {
    # Check if the user is in $DIRMNT
    INMNT=$(echo $PWD | grep $DIRMNT | wc -l)
    echo $INMNT
}

func runcmd(){
    if [[ $(checkmnt) -gt 0 ]]; then
        if [[ $(checkinmnt) -gt 0 ]]; then
            REMOTEDIR=$(echo $PWD | sed "s,$DIRMNT,,")
            echo $REMOTEDIR
            ssh $HOST "source ~/.zshrc && conda activate pytorch && cd $REMOTEDIR && ($CMD)"
        fi
    fi
}


if [[ -z $DIR ]]; then
    DIR='desk://'
fi

if [[ $MOUNT -eq 1 ]]; then
    echo "Will mount $DIR in $DIRMNT"
    sshfs $DIR $DIRMNT
fi

if [[ $UNMOUNT -eq 1 ]]; then
    echo "Will unmount $DIRMNT"
    fusermount -u $DIRMNT
fi

if [[ $(checkmnt) -gt 0 ]]; then
    echo "$DIRMNT is mounted" 
fi

if [[ ! -z $CMD ]]; then
    runcmd
fi
