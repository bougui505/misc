#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Apr 28 13:51:22 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

MAESTROPATH=/pasteur/appa/scratch/bougui/vortex

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

push:
	ssh -t maestro << EOF
	mkdir -p ${MAESTROPATH}
	EOF
	rsync -a --update -P -h . maestro:${MAESTROPATH}/

build: push
	ssh -t maestro << EOF
	cd ${MAESTROPATH}
	sbatch vortex.slurm.sh
	EOF

pull:
	rsync -a --update -P -h maestro:${MAESTROPATH}/ .
