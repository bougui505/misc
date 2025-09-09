#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Sep  8 16:01:49 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

# Defines the temporary directory for Singularity/Apptainer builds,
# using a path relative to the Makefile's location.
SINGULARITY_TMP_DIR := $(patsubst %/,%,$(dir $(realpath $(firstword $(MAKEFILE_LIST)))))/tmp

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

LAST_PREREQUISITE := $(lastword $^)

bougui_1.sif: bougui_1.def
	mkdir -p $(SINGULARITY_TMP_DIR)
	APPTAINER_TMPDIR=$(SINGULARITY_TMP_DIR) apptainer build --force $@ $<

bougui_2.sif: bougui_2.def bougui_1.sif
	mkdir -p $(SINGULARITY_TMP_DIR)
	APPTAINER_TMPDIR=$(SINGULARITY_TMP_DIR) apptainer build --force $@ $<

bougui.sif: bougui_1.sif
	cd ..
	[[ -h $@ ]] && rm -v $@
	ln -vs bougui/$< $@

clean:
	rm -v bougui_1.sif
