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

REMOTEHOST=horace
REMOTEPATH=/home/bougui/source/misc/singularity/openmm

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

openmm.sif: openmm.def
	singularity build --force $@ $^
	# rsync -a --update -P -h $^ ${REMOTEHOST}:${REMOTEPATH}/.
	# ssh -t ${REMOTEHOST} << EOF
	# cd ${REMOTEPATH}
	# pwd
	# singularity build --force $@ $^
	# EOF
	# rsync -a --update -P -h ${REMOTEHOST}:${REMOTEPATH}/$@ .

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
	rsync -a --update -P -h $^ openmm.def desk:/c7/scratch2/bougui/source/misc/singularity/openmm/
