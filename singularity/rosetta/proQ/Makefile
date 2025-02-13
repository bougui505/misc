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

all: bin/run_all_external.pl

test: all
	./bin/run_all_external.pl -pdb structures/1ycr.pdb

bin/aa321CA.pl:
	mkdir -p bin
	wget https://github.com/bjornwallner/ProQ_scripts/raw/refs/heads/master/bin/aa321CA.pl -O $@
	chmod +x $@

bin/create_profile.sh:
	mkdir -p bin
	wget https://github.com/bjornwallner/ProQ_scripts/raw/refs/heads/master/bin/create_profile.sh -O $@
	chmod +x $@

bin/run_all_external.pl: bin/aa321CA.pl bin/create_profile.sh
	mkdir -p bin
	wget https://github.com/bjornwallner/ProQ_scripts/raw/refs/heads/master/bin/run_all_external.pl -O $@
	sed -i 's,/proj/wallner/users/x_bjowa/DB,/empohrr/home/areguei/Téléchargements,' bin/run_all_external.pl
	chmod +x $@
