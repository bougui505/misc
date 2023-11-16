#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-06-05 15:33:23 (UTC+0200)

DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
if [ -p /dev/stdin ]; then
    # echo 'Pipe'
    "$DIRSCRIPT/np.py" "-c" "$1"
else
    # echo 'No pipe'
    "$DIRSCRIPT/np.py" "--nopipe" "-c" "$1"
fi
