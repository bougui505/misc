#!/usr/bin/env zsh
# shellcheck shell=bash
# -*- coding: UTF8 -*-

set -e  # exit on error
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
singularity exec --nv $DIRSCRIPT/nvitop.sif nvitop
