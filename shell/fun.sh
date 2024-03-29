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
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Create a fun.sh file to write functions to be sourced in the current working directory
    -h, --help print this help message and exit
    -l, --list list functions in the current fun.sh file
EOF
}



list_fun () {
    # bash -c '. fun.sh; typeset -F'
    echo $FUNFILES \
        | awk -F":" '{for (i=1;i<=NF;i++){print $i}}' | xargs $DIRSCRIPT/list_fun.sh | sort -u
    exit 0
}

EDIT=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -l|--list) list_fun ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

create_file () {
    if [ ! -f fun.sh ]; then
        cat << EOFF > fun.sh
#!/usr/bin/env bash

Cyan='\033[0;36m'
NC='\033[0m' # No Color

echo -e "\$Cyan"
cat << EOF
\$(date): sourcing \$0
EOF
echo -e \$FUNFILES\$NC

MYTMP=\$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap '/bin/rm -rf "\$MYTMP"' EXIT INT  # Will be removed at the end of the script

[ -z \$FUNFILES ] && export FUNFILES=""
precmd() {
    [ -f fun.sh ] && source fun.sh && FUNFILES+="\$(realpath fun.sh)" && FUNFILES=\$(echo \$FUNFILES | awk -F":" '{for (i=1;i<=NF;i++){print \$i}}' | sort -u | awk '{printf \$1":"}')
}

func1 () {
    echo
}
EOFF
    fi
    nvim fun.sh || vim fun.sh
}

create_file
cat << EOF
#################################################
run one of the following command:

source fun.sh
. fun.sh
sf

to load the functions
#################################################
EOF
