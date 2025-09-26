#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Sep 24 14:55:33 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

HORACEDIR=/horace/bougui/colabfold

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

colabfold_1.5.5-cuda12.2.2.sif:
	singularity pull docker://ghcr.io/sokrypton/colabfold:1.5.5-cuda12.2.2

test-msa: colabfold_1.5.5-cuda12.2.2.sif
	./colabfold_1.5.5-cuda12.2.2.sif colabfold_batch data/1ycr_A.fa out_dir --msa-only

push_colabfold_1.5.5-cuda12.2.2.sif: colabfold_1.5.5-cuda12.2.2.sif
	ssh horace << EOF
		mkdir -p ${HORACEDIR}
	EOF
	rsync -a --update -P -h $< horace:${HORACEDIR}/.

setup_db:
	ssh horace << EOF
		mkdir -p ${HORACEDIR}
		cd ${HORACEDIR}
		[ ! -f setup_databases.sh ] && wget https://raw.githubusercontent.com/sokrypton/ColabFold/refs/heads/main/setup_databases.sh && sed -i '2i\MMSEQS_NO_INDEX=1' setup_databases.sh
		chmod u+x setup_databases.sh
		mkdir -p DB \
			&& tsp ./colabfold_1.5.5-cuda12.2.2.sif ./setup_databases.sh ${HORACEDIR}/DB
	EOF
