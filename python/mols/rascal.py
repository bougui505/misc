#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Feb  7 15:26:27 2024

# RASCAL: RApid Similarity CALculation

from rdkit import Chem
from rdkit.Chem import rdRascalMCES


def rascal(smi1, smi2):
    """
    >>> smi1 = 'Oc1cccc2C(=O)C=CC(=O)c12'
    >>> smi2 = 'O1C(=O)C=Cc2cc(OC)c(O)cc12'
    >>> sim = rascal(smi1, smi2)
    >>> sim
    0.36909323116219667
    """
    mol1 = Chem.MolFromSmiles(smi1)
    mol2 = Chem.MolFromSmiles(smi2)
    opts = rdRascalMCES.RascalOptions()
    opts.similarityThreshold = 0.
    results = rdRascalMCES.FindMCES(mol1, mol2, opts)
    n = len(results)
    if n == 0:
        return 0.0
    sim = sum(r.similarity for r in results)/n
    return sim

import os

if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--smi1", help='First SMILES string', metavar='Oc1cccc2C(=O)C=CC(=O)c12')
    parser.add_argument("--smi2", help='Second SMILES string', metavar='O1C(=O)C=Cc2cc(OC)c(O)cc12')
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()
    if args.smi1 is not None and args.smi2 is not None:
        sim = rascal(args.smi1, args.smi2)
        print(f"{sim:.2g}")
