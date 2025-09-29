#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Sep 29 11:41:42 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

simplefold.sif: simplefold.def
	apptainer build --force $@ $<

data/1ycr.fasta:
	mkdir -p $(@D)
	pdbsumup --pdb 1ycr --seq \
		| recawk '{if (rec["chain"]=="A" || rec["chain"]=="B"){printf "> chain "rec["chain"]"\n"rec["sequence"]"\n"}}' \
		> $@

data/1ycr_A.fasta:
	mkdir -p $(@D)
	pdbsumup --pdb 1ycr --seq \
		| recawk '{if (rec["chain"]=="A"){printf "> chain "rec["chain"]"\n"rec["sequence"]"\n"}}' \
		> $@

configs:
	git clone git@github.com:apple/ml-simplefold.git
	ln -s ml-simplefold/configs

test_1ycr_A: configs
	apptainer run simplefold.sif simplefold \
		--simplefold_model simplefold_100M \
		--num_steps 500 --tau 0.6 \
		--nsample_per_protein 10 \
		--plddt \
		--fasta_path data/1ycr_A.fasta \
		--output_dir out \
		--backend torch
