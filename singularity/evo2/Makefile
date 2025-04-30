#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Apr 30 08:53:38 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

REMOTEHOST=horace
REMOTEDIR=/home/bougui/source/misc/singularity/evo2

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

push:
	ssh ${REMOTEHOST} "mkdir -p ${REMOTEDIR}"
	rsync -a --update -P -h . ${REMOTEHOST}:${REMOTEDIR}/

build: push evo2.def
	ssh ${REMOTEHOST} << EOF
		cd ${REMOTEDIR}
		apptainer build --force evo2.sif evo2.def
	EOF

pull:
	rsync -a --update -P -h ${REMOTEHOST}:${REMOTEDIR}/ .
