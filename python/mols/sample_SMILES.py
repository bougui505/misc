#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jun 10 09:23:18 2025

"""
To generate all possible SMILES strings that represent the same molecule (i.e., all valid SMILES encodings for the same molecular structure) using RDKit, you need to enumerate all possible atom orderings and generate the SMILES for each. RDKit provides a way to randomize the atom ordering and output canonical or non-canonical SMILES.
"""

import sys

from rdkit import Chem


def enumerate_smiles(smiles, num=1000):
    mol = Chem.MolFromSmiles(smiles)
    smiles_set = set()
    for _ in range(num):
        smi = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
        smiles_set.add(smi)
    return list(smiles_set)

# Example usage
# original_smiles = "NC(CCC(=O)NC(CSCC(O)CCc1ccccc1)C(=O)N=CC(=O)O)C(=O)O"
# original_smiles = "OC(=O)c1ccc(cc1)c2nn(C(=O)c3c(Cl)cccc3C(F)(F)F)c4ccccc24"
original_smiles = sys.argv[1]
all_smiles = enumerate_smiles(original_smiles, num=2000)
print(all_smiles)
print(f"Generated {len(all_smiles)} unique SMILES.")
