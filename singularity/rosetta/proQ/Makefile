#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Feb 13 16:19:41 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

INSTALL_DIR="./scripts/"

scripts/aa321CA.pl:
	wget https://github.com/bjornwallner/ProQ_scripts/raw/refs/heads/master/bin/aa321CA.pl -O $@
	chmod +x $@

scripts/run_all_external.pl: scripts/aa321CA.pl
	wget https://github.com/bjornwallner/ProQ_scripts/raw/refs/heads/master/bin/run_all_external.pl -O $@
	sed "s,\$$seq=`\$$INSTALL_DIR/bin/aa321CA.pl \$$pdb`;,\$$seq=`${INSTALL_DIR}/aa321CA.pl \$$pdb`;"
	chmod +x $@
