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
trap "tsp -k $TSPID" EXIT INT

PROMPT="$1"
if [ ! -t 0 ]; then
  PROMPT+="\n$(</dev/stdin)"
fi

TSPID=$(tsp $DIRSCRIPT/server.sh)
sleep 0.5

# $DIRSCRIPT/ollama run llama2 "$PROMPT"
# $DIRSCRIPT/ollama run mistral "$PROMPT"
$DIRSCRIPT/ollama run codellama "$PROMPT" --nowordwrap
