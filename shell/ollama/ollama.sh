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

OLLAMA_OUPUT_DIR="$HOME/Documents/ollama-ouputs"
PROMPT="$1"
OUTFILENAME="$OLLAMA_OUPUT_DIR/$(echo $PROMPT | sed 's/ /_/g ; s/?//g').md"

if [[ -f $OUTFILENAME ]]; then
  batcat -p $OUTFILENAME
  echo "ANSWER READ FROM CACHE: $OUTFILENAME"
  exit 0
fi

tsp -C

mkdir -p $OLLAMA_OUPUT_DIR

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
trap "tsp -k $TSPID && (cat $OUTFILENAME | xclip) && echo 'ANSWER COPIED TO CLIPBOARD'" EXIT INT

if [ ! -t 0 ]; then
  PROMPT+="\n$(</dev/stdin)"
fi

TSPID=$(tsp $DIRSCRIPT/server.sh)
sleep 0.5

# $DIRSCRIPT/ollama run llama2 "$PROMPT"
$DIRSCRIPT/ollama run custom "$PROMPT" --nowordwrap | tee $OUTFILENAME
# $DIRSCRIPT/ollama run codellama "$PROMPT"
