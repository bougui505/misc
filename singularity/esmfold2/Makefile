SHELL := zsh
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

.PHONY: all clean

all: esmfold2.sif

esmfold2.sif: esmfold2.def
	mkdir -p /scratch/arcturus/tmp
	APPTAINER_TMPDIR=/scratch/arcturus/tmp apptainer build --force $@ $<

clean:
	rm -f esmfold2.sif
