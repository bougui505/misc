#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Aug 22 14:23:58 2024

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: help test #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

openmm.sif: openmm.def
	singularity build --force $@ $^

openmm-pip.sif: openmm-pip.def
	singularity build --force $@ $^

test-openmm: openmm.sif
	singularity exec $^ python3 -m openmm.testInstallation

test-openmm-pip: openmm-pip.sif
	singularity exec $^ python3 -m openmm.testInstallation

push: openmm.sif
	ssh desk << EOF
	mkdir -p /c7/scratch2/bougui/source/misc/singularity/openmm
	EOF
	rsync -a --update -P -h $^ desk:/c7/scratch2/bougui/source/misc/singularity/openmm/
