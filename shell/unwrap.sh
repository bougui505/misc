#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Aug 12 11:38:23 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

awk '{
  if (NF>10 && $1!="-"){  # more than 10 words in the line and not starting by "-"
    for (i=1;i<=NF;i++){  # unwrap the line
      printf $i" "
    }
  }
  else{
    print  # print the line as is
  }
  }' $1 # \
  # | sponge $1
