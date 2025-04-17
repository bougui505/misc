#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Apr 17 14:11:08 2025

import numpy as np
import typer
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS, rdMolAlign

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.callback()
def callback(debug:bool=False):
    """
    This is a template file for a Python script using Typer.
    It contains a main function and a test function.
    """
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = debug


def get_rmsd(mol1, mol2, maxIters):
    mol1 = Chem.RemoveHs(mol1)
    mol2 = Chem.RemoveHs(mol2)
    mcs = rdFMCS.FindMCS([mol1, mol2])
    numAtoms = mcs.numAtoms
    print(f"numAtoms={numAtoms}")
    patt = Chem.MolFromSmarts(mcs.smartsString)
    refMatch = mol1.GetSubstructMatch(patt)
    mv = mol2.GetSubstructMatch(patt)
    # rms = AllChem.AlignMol(mol2,mol1,atomMap=list(zip(mv,refMatch)))
    # get coordinates of the matching atoms
    conf1 = mol1.GetConformer()
    conf2 = mol2.GetConformer()
    coords1 = [conf1.GetAtomPosition(i) for i in refMatch]
    coords2 = [conf2.GetAtomPosition(i) for i in mv]
    # compute the RMSD
    rmsd = 0
    for i in range(len(refMatch)):
        xyz1 = np.asarray([coords1[i].x, coords1[i].y, coords1[i].z])
        xyz2 = np.asarray([coords2[i].x, coords2[i].y, coords2[i].z])
        rmsd += ((xyz1 - xyz2) ** 2).sum()
    rmsd = (rmsd / len(refMatch)) ** 0.5
    # rmsd = rdMolAlign.AlignMol(mol2, mol1, atomMap=list(zip(mv, refMatch)), maxIters=maxIters)
    return rmsd


@app.command()
def mcs_rmsd(
    sdf1:str,
    sdf2:str,
    maxIters:int=0,
):
    """
    Compute the RMSD between two molecules using the MCS
    :param sdf1: path to the first sdf file
    :param sdf2: path to the second sdf file
    :param maxIters: maximum number of iterations for the alignment (default=0, no alignment)
    """
    mol1 = Chem.MolFromMolFile(sdf1)
    mol2 = Chem.MolFromMolFile(sdf2)
    rmsd = get_rmsd(mol1, mol2, maxIters)
    print(f"{rmsd=}")
    return rmsd

@app.command()
def mcs_rmsd_pairwise(
    sdf:str,
):
    # iterate over all pairs of molecules in the sdf file
    for i, mol1 in enumerate(Chem.SDMolSupplier(sdf)):
        for j, mol2 in enumerate(Chem.SDMolSupplier(sdf)):
            name1 = mol1.GetProp("_Name")
            name2 = mol2.GetProp("_Name")
            print(f"{name1=}")
            print(f"{name2=}")
            rmsd = get_rmsd(mol1, mol2, 0)
            print(f"{rmsd=}")
            print("--")




if __name__ == "__main__":
    import doctest
    import sys

    @app.command()
    def test():
        """
        Test the code
        """
        doctest.testmod(
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF
        )

    @app.command()
    def test_func(func:str):
        """
        Test the given function
        """
        print(f"Testing {func}")
        f = getattr(sys.modules[__name__], func)
        doctest.run_docstring_examples(
            f,
            globals(),
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF,
        )

    app()
