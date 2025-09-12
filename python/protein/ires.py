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
import sys
from pymol2 import PyMOL
import numpy as np

PDB_DIR = os.path.expanduser("~/pdb")

import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

def loader(pdb, pml, selection=None):
    """
    Loads a PDB file or fetches it from the PDB database.
    Optionally removes atoms not matching the selection.
    """
    if os.path.isfile(pdb):
        pml.cmd.load(pdb)
    else:
        pml.cmd.fetch(pdb, path=PDB_DIR)
    if selection is not None:
        pml.cmd.remove(f"not {selection}")

def print_deltasasa(model, sasas, filter_sasa=True):
    """
    Prints residues with positive delta SASA values.
    """
    assert len(model.atom) == len(sasas)
    for i, atom in enumerate(model.atom):
        delta_sasa = sasas[i]
        if atom.name == "CA":
            if filter_sasa:
                doprint=False
                if delta_sasa > 0.0:
                    doprint=True
            else:
                doprint=True
            if doprint:
                # AI! convert the 3 letters-code resn to 1 letter code
                resn = atom.resn
                resi = atom.resi
                chain = atom.chain
                print(f"{resn=!s}")
                print(f"{resi=!s}")
                print(f"{chain=!s}")
                print(f"{delta_sasa=:.4f}")
                print("--")


@app.command()
def get_interface(
    pdb: str = typer.Argument(..., help="PDB ID or path to a PDB file."),
    filter_sasa: bool = typer.Option(True, help="Only print residues with positive delta SASA values."),
):
    """
    Computes and prints the interface residues of a protein.

    For each chain, the relative SASA (rSASA) is computed for the entire protein
    and then for the isolated chain. The difference (delta SASA) indicates
    residues that become more exposed when other chains are removed,
    suggesting they are part of the interface.
    """
    with PyMOL() as pml:
        loader(pdb, pml)
        pml.cmd.get_sasa_relative()  # compute the relative SASA and store it in the b-factor (value between 0.0 (fully buried) and 1.0 (fully exposed))
        model_ref = pml.cmd.get_model()
        chains = list(set(atom.chain for atom in model_ref.atom))
        chains.sort()
        print(f"{chains=}")
        # Get reference model individually by chain
        chain_models_ref = dict()
        for chain in chains:
            chain_models_ref[chain] = pml.cmd.get_model(f"chain {chain}")
    for chain in chains:
        with PyMOL() as pml:
            loader(pdb, pml, selection=f"chain {chain}")
            pml.cmd.get_sasa_relative()  # compute the relative SASA and store it in the b-factor
            chain_model = pml.cmd.get_model()
            chain_model_ref = chain_models_ref[chain]
            # compute the difference of rSASA between chain_model and model
            # positive values should be at the interface between the current chain and the other ones.
            sasa = np.asarray([atom.b for atom in chain_model.atom])
            sasa_ref = np.asarray([atom.b for atom in chain_model_ref.atom])
            delta_sasa = sasa - sasa_ref
            print_deltasasa(chain_model_ref, delta_sasa, filter_sasa)


if __name__ == "__main__":
    app()

