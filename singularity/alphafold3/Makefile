#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jan 24 13:58:55 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

REMOTEDIR=/pasteur/appa/homes/bougui/alphafold3

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

alphafold3.sif: alphafold3.def
	apptainer build $@ $^

push:
	ssh maestro << EOF
	mkdir -p ${REMOTEDIR}
	EOF
	rsync -a --update -P -h . maestro:${REMOTEDIR}/

pull:
	rsync -a --update -P -h maestro:${REMOTEDIR}/ .

clean:
	/bin/rm -vf slurm-*.out || echo "Nothing to clean"
	ssh maestro << EOF
	cd ${REMOTEDIR} \
		&& /bin/rm -vf slurm-*.out
	EOF

test: push
	ssh maestro << EOF
	date
	cd ${REMOTEDIR}
	pwd
	sbatch ./alphafold3-predict.sh --json_path=data/fold_input.json --output_dir=test_output
	EOF
