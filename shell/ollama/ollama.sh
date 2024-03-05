#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Feb 29 15:04:17 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

PROMPT="$1"
if [ ! -t 0 ]; then
  PROMPT+="\n$(</dev/stdin)"
fi

if tsp | grep $DIRSCRIPT/server.sh > /dev/null ; then
  echo "$DIRSCRIPT/server.sh running" > /dev/null
else
  tsp -S 2
  tsp $DIRSCRIPT/server.sh
  sleep 2
fi

# $DIRSCRIPT/ollama run llama2 "$PROMPT"
$DIRSCRIPT/ollama run mistral "$PROMPT"
# $DIRSCRIPT/ollama run orca-mini "$PROMPT"
