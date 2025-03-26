#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Mar 26 18:23:51 2025

trap "tsp -k $TSPID" EXIT INT  # Will be removed at the end of the script

TSPID=$(tsp ollama serve)
sleep 0.5
# ollama pull deepseek-coder-v2
sgpt --model ollama/deepseek-coder-v2 $@
