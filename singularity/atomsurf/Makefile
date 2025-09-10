#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Sep 10 10:03:36 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

atomsurf.sif: atomsurf.def
	mkdir -p /tmp/apptainer/atomsurf
	APPTAINER_TMPDIR=/tmp/apptainer/atomsurf apptainer build --force $@ $< 2>&1 | tee atomsurf.log
	rm -rf atomsurf  # remove the git repo
	rm -rf /tmp/apptainer/atomsurf
	rm atomsurf.log
