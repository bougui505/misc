#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Sep  4 16:37:39 2024

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

cuda-repo-debian12-12-6-local_12.6.1-560.35.03-1_amd64.deb:
	[[ -f cuda-repo-debian12-12-6-local_12.6.1-560.35.03-1_amd64.deb ]] && rm -v cuda-repo-debian12-12-6-local_12.6.1-560.35.03-1_amd64.deb
	wget https://developer.download.nvidia.com/compute/cuda/12.6.1/local_installers/cuda-repo-debian12-12-6-local_12.6.1-560.35.03-1_amd64.deb
	touch cuda-repo-debian12-12-6-local_12.6.1-560.35.03-1_amd64.deb

cuda-repo-debian11-12-6-local_12.6.0-560.28.03-1_amd64.deb:
	[[ -f cuda-repo-debian11-12-6-local_12.6.0-560.28.03-1_amd64.deb ]] && rm -v cuda-repo-debian11-12-6-local_12.6.0-560.28.03-1_amd64.deb
	wget https://developer.download.nvidia.com/compute/cuda/12.6.0/local_installers/cuda-repo-debian11-12-6-local_12.6.0-560.28.03-1_amd64.deb
	touch cuda-repo-debian11-12-6-local_12.6.0-560.28.03-1_amd64.deb

gnina.sif: gnina.def
	singularity build --nv --force gnina.sif gnina.def
