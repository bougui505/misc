#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Feb 13 09:28:30 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

rosetta.tar.gz:
	git clone https://github.com/RosettaCommons/rosetta
	tar -cvf - rosetta | pigz > $@
	rm -rf rosetta/

rosetta.sif: rosetta.def rosetta.tar.gz
	apptainer build --force $@ rosetta.def
