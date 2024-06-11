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

molref = Chem.MolFromMol2File('NCT.mol2', sanitize=True)
suppl = Chem.SDMolSupplier('6CNK_ECD_A4A4_noNCT_Ligands.sdf', sanitize=True)
# suppl = Chem.SDMolSupplier('Nico3bp.sdf', sanitize=True)


def get_coords(mol):
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        # print(atom.GetSymbol(), positions.x, positions.y, positions.z)
        coords.append([positions.x, positions.y, positions.z])
    coords = np.asarray(coords)
    return coords

coord_ref = get_coords(molref)


for mol in suppl:
    coords = get_coords(mol)
    match = mol.GetSubstructMatch(molref, useChirality=False)
    nmatch = len(match)
    if nmatch > 0:
        rmsd = np.sqrt(((coords[match, :] - coord_ref[:nmatch])**2).mean(axis=0).sum())
        print(mol.GetProp("_Name"), nmatch, rmsd)
    else:
        print(mol.GetProp("_Name"), nmatch, 9999.99)
