#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jan 22 08:22:14 2026

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

dos2unix \
  | awk '
BEGIN{
  sequence=""
}
{
  if ($0 ~ /^>/){
    if (sequence != ""){
      print("lab="label)
      print("seq="sequence)
      print("--")
    }
    label=$0
    sequence=""
  }
  else{
    sequence = sequence $0
  }
}'
