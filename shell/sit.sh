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
# sit comes from Script IT or Save IT ;-) ...

set -e  # exit on error
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
SITDIR='cmds'

function usage () {
    cat << EOF
Store the given command after '--' in an executable sh script file stored in scripts
    sit -n test -- ls -a

The given command given must escape shell command delimiters such as && ; || | ...
If such delimiters are present git the command with double quote
For example:
    sit -n test -- "pwd && ls -a"
Or escape the special characters:
    sit -n test -- pwd \\&\\& ls -a

The commands are stored in the cmds directory. The files are executable zsh files.
They can be run as a shell script:
    ./cmds/test.sh
However the user specific shell environment is not seen. To run in the current shell please use source command:
    source cmds/test.sh
or
    . cmds/test.sh

    -h, --help print this help message and exit
    -n, --name name of the sit file without extension
    -s, --search search in scripts directory
    -e, --edit display the selected sit;
               the user can edit the displayed command
               and the sit will be saved back to the sit file
EOF
}

if [ "$#" -eq 0 ]; then
    usage
    exit 1
fi

save_sit () {
    echo "#!/usr/bin/env zsh\n$CMD" > $SITDIR/$NAME.sh
    chmod u+x $SITDIR/$NAME.sh
}


SEARCH=0
EDIT=0
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--name) NAME="$2"; shift;;
        -s|--search) SEARCH=1;;
        -e|--edit) EDIT=1;SEARCH=1;;
        -h|--help) usage; exit 0 ;;
        --) CMD="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

if [ $SEARCH -eq 1 ]; then
    # CMD=$(ls $SITDIR/*.sh | fzf)
    SITFILE=$(ls $SITDIR/* | fzf --preview='less {}')
    if [ $EDIT -eq 1 ]; then
        nvim $SITFILE
    fi
    CMD=$(grep -v "^#!" $SITFILE | sed '/^$/d' | tr '\n' ';')
    # stty -echo is to prevent xdotool to write cmd in terminal before typing (see: https://stackoverflow.com/a/35976098/1679629)
    stty -echo && xdotool type $CMD && stty echo
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
save_sit
