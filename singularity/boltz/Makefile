#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Dec  9 11:44:14 2024

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

boltz.sif: boltz.def
	singularity build -F $@ $^

test:
	./boltz-predict.sh data/1t4e.yml --use_msa_server --override

test2:
	./boltz-predict.sh data/1t4e.fasta --use_msa_server --override
