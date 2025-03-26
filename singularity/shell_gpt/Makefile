#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Mar 26 18:19:06 2025

SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

outputs=target1 target2
.PHONY: help clean #  which targets are not represented by files

help:
	@echo "\e[4mTargets:\e[0m"
	@grep '^[[:alnum:]].*:' Makefile

shell_gpt.sif: shell_gpt.def
	apptainer build $^ $@ || echo "done"
	cat << EOF >! ~/.config/shell_gpt/.sgptrc
	DEFAULT_MODEL=ollama/your_local_model
	OPENAI_USE_FUNCTIONS=false
	USE_LITELLM=true
	OPENAI_API_KEY=any_value
	EOF
