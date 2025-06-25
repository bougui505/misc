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

# boltz.sif: boltz.def
# 	singularity build -F $@ $^

# rsync-boltz.sif:
# 	rsync -a --update -P -h boltz.sif desk:/c7/scratch2/bougui/source/misc/singularity/.

# boltz_data/ccd.pkl:
# 	sing ./modified_residues.py -s "CCCCCCCCCCCCCCCC(=O)SCCNC(=O)CCNC(=O)C(O)C(C)(C)COP(O)(=O)OCC(N)C(O)=O" -a SER -o 4HH_G9S
# 	sing ./modified_residues.py -s "CC1=NC=C(COP(O)(O)=O)C(CNCCCCC(N)C(O)=O)=C1O" -a LYS -o KPLP

test:
	./boltz-predict.sh data/1t4e.yml --use_msa_server --override

test2:
	./boltz-predict.sh data/1t4e.fasta --use_msa_server --override

boltz_data/ccd.pkl:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl -O $@

boltz_data/boltz1_conf.ckpt:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt -O $@

boltz_data/boltz2_conf.ckpt:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt -O $@

boltz_data/boltz2_aff.ckpt:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt -O $@

boltz.sif: boltz.def boltz_data/ccd.pkl boltz_data/boltz1_conf.ckpt boltz_data/boltz2_conf.ckpt boltz_data/boltz2_aff.ckpt
	singularity build -F $@ $<
