#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Aug 21 16:34:41 2024

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help clean test #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

bougui.sif: bougui.def
	singularity build --force bougui.sif bougui.def

bougui.sif-maestro:
	rsync -a --update -P -h chimera maestro:/pasteur/appa/homes/bougui/source/misc/singularity/
	ssh maestro << EOF
	cd /pasteur/appa/homes/bougui/source/misc/singularity
	git pull
	module load apptainer
	apptainer build --force bougui.sif bougui.def
	EOF

test-cuda: bougui.sif
	singularity run --nv bougui.sif python3 -c 'import torch; print(torch.cuda.is_available())'
