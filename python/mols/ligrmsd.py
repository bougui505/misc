#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jun 11 14:18:45 2024

import numpy as np
from rdkit import Chem


def get_coords(mol):
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        # print(atom.GetSymbol(), positions.x, positions.y, positions.z)
        coords.append([positions.x, positions.y, positions.z])
    coords = np.asarray(coords)
    return coords

def get_rmsd(mol, molref, coord_ref):
    coords = get_coords(mol)
    match = mol.GetSubstructMatch(molref, useChirality=False)
    nmatch = len(match)
    match_ratio = nmatch / len(coord_ref)
    if nmatch > 0:
        rmsd = np.sqrt(((coords[match, :] - coord_ref[:nmatch])**2).mean(axis=0).sum())
        # print(mol.GetProp("_Name"), nmatch, rmsd)
    else:
        rmsd = 9999.99
        # print(mol.GetProp("_Name"), nmatch, 9999.99)
    molname = mol.GetProp("_Name")
    return rmsd, match_ratio, molname

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mol2ref", help="Reference mol2 file")
    parser.add_argument("--sdf", help="sdf file with molecules to compare with references")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    molref = Chem.MolFromMol2File(args.mol2, sanitize=True)
    suppl = Chem.SDMolSupplier(args.sdf, sanitize=True)

    coord_ref = get_coords(molref)

    for mol in suppl:
        rmsd, match_ratio, molname = get_rmsd(mol, molref, coord_ref)
        print(f"{molname=}")
        print(f"{match_ratio=:.3g}")
        print(f"{rmsd=:.4g}")
        print("--")
