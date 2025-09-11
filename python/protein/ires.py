#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Sep 11 09:47:28 2025

import typer
import os
from pymol2 import PyMOL

PDB_DIR = os.path.expanduser("~/pdb")

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.command()
def get_interface(
    pdb:str
    ):
    with PyMOL() as pml:
        if os.path.isfile(pdb):
            pml.cmd.load(pdb)
        else:
            pml.cmd.fetch(pdb, path=PDB_DIR)
        model = pml.cmd.get_model()
        chains = set(atom.chain for atom in model.atom)
        print(f"Chains in the model: {sorted(list(chains))}")

if __name__ == "__main__":
    app()

