#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
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

# Create a zsh script file from the given command
# The file is created in a scripts directory
# sit comes from Script IT

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
SITDIR='scripts'

function usage () {
    cat << EOF
Store the given command after '--' in an executable sh script file stored in scripts
E.g.: sit -n test -- ls -a

    -h, --help print this help message and exit
    -n, --name name of the sit file without extension
    -s, --search search in scripts directory
EOF
}

if [ "$#" -eq 0 ]; then
    usage
    exit 1
fi

SEARCH=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--name) NAME="$2"; shift;;
        -s|--search) SEARCH=1;;
        -h|--help) usage; exit 0 ;;
        --) CMD="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

if [ $SEARCH -eq 1 ]; then
    # CMD=$(ls $SITDIR/*.sh | fzf)
    for FILE in $(ls $SITDIR/*.sh); do
        CONTENT=$(grep -v '^#!' $FILE)
        echo "$FILE $CONTENT"
    done \
        | fzf \
        | awk '{print $1}' \
        | read CMD
    eval $CMD
    exit 0
fi

if [ -z $CMD ]; then
    echo "Give a command to store after the '--' :"
    echo "sit -n test -- ls -a\n"
    usage
    exit 1
fi
if [ -z $NAME ]; then
    echo "-n, --name is mandatory\n"
    usage
    exit 1
fi

if [ ! -d $SITDIR ]; then
    mkdir -v $SITDIR
fi
echo "#!/usr/bin/env zsh\n$CMD" > $SITDIR/$NAME.sh
chmod u+x $SITDIR/$NAME.sh
