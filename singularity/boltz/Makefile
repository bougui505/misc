#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Dec  9 11:44:14 2024

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

boltz_data/ccd.pkl:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-1/resolve/main/ccd.pkl -O $@

boltz_data/mols:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-2/resolve/main/mols.tar -O - | tar -x -C $(@D)

boltz_data/boltz1_conf.ckpt:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-1/resolve/main/boltz1_conf.ckpt -O $@

boltz_data/boltz2_conf.ckpt:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_conf.ckpt -O $@

boltz_data/boltz2_aff.ckpt:
	mkdir -p $(@D)
	wget https://huggingface.co/boltz-community/boltz-2/resolve/main/boltz2_aff.ckpt -O $@

boltz.sif: boltz.def boltz_data/ccd.pkl boltz_data/mols boltz_data/boltz1_conf.ckpt boltz_data/boltz2_conf.ckpt boltz_data/boltz2_aff.ckpt
	mkdir -p tmp
	APPTAINER_TMPDIR=$$(pwd)/tmp singularity build -F $@ $<

boltz-dgx.sif: boltz2-dgx.def boltz_data/ccd.pkl boltz_data/mols boltz_data/boltz1_conf.ckpt boltz_data/boltz2_conf.ckpt boltz_data/boltz2_aff.ckpt
	mkdir -p tmp
	APPTAINER_TMPDIR=$$(pwd)/tmp singularity build -F $@ $<

boltz.sif-maestro: boltz.def boltz_data/ccd.pkl boltz_data/mols boltz_data/boltz1_conf.ckpt boltz_data/
	ssh maestro << EOF
	cd /pasteur/appa/homes/bougui/source/misc
	git pull
	cd /pasteur/appa/homes/bougui/source/misc/singularity/boltz
	module load apptainer
	make boltz.sif
	EOF

test:
	apptainer run --nv boltz.sif boltz predict data/1t4e.yml --use_msa_server --override

test-cpu:
	apptainer run boltz.sif boltz predict data/1t4e.yml --use_msa_server --override --accelerator cpu
