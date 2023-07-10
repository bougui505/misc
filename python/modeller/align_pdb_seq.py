#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2016-09-06 11:52:25 (UTC+0200)

import modeller as mod
import os
import sys


def main(pdbs, out=sys.stdout):
    """
    Align the sequence of the given pdb file names.
    • pdbs: list of pdb file names
    """
    env = mod.environ()
    mdl = mod.model(env)
    aln = mod.alignment(env)
    codes = []
    for pdb in pdbs:
        mdl.read(pdb)
        code = os.path.splitext(pdb)[0]
        codes.append(code)
        aln.append_model(mdl, atom_files=code, align_codes=code)
    aln.salign()
    try:  # if outfile is a filename
        with open(out) as outfile:
            aln.write(file=outfile, alignment_format="PAP")
    except TypeError:  # Outfile is already a file object (e.g. sys.stdout)
        aln.write(file=out, alignment_format="PIR")


if __name__ == "__main__":
    PDBS = sys.argv[1:]  # list of pdb file names
    main(PDBS)
