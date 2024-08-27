#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Aug 27 10:06:19 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

# sudo multitail -i /root/.local/share/backintime/worker.progress -i /root/.local/share/backintime/worker.message
# sudo tail -f /root/.local/share/backintime/worker.progress 2> /dev/null
sudo watch -d cat /root/.local/share/backintime/worker.progress =(echo "--") /root/.local/share/backintime/worker.message
