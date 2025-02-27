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

.PHONY: help #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

openmm.sif: openmm.def
	singularity build --force $@ $^

plumed.sif: plumed.def openmm.sif
	P1=$$(echo $^ | awk '{print $$1}')
	P2=$$(echo $^ | awk '{print $$2}')
	apptainer build --force $@ $$P1

openmm-plumed.sif: openmm-plumed.def plumed.sif
	P1=$$(echo $^ | awk '{print $$1}')
	P2=$$(echo $^ | awk '{print $$2}')
	apptainer build --force $@ $$P1

test-openmm-plumed: openmm-plumed.sif
	./openmm-plumed.sif python -c "from openmmplumed import PlumedForce"

test-openmm: openmm.sif
	singularity exec $^ python3 -m openmm.testInstallation
