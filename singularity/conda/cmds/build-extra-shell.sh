#!/usr/bin/env zsh
sudo singularity build --force --sandbox extra.{sif,def} && sudo singularity build --force shell.{sif,def} && rsync shell.sif desk:~/source/misc/singularity/conda
