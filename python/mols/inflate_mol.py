#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-12-18 11:15:44 (UTC+0100)

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize
import os
import sys


def fixmol(mol):
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    AllChem.MMFFOptimizeMolecule(mol)
    return mol


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-m', '--mol')
    parser.add_argument('-s', '--smiles')
    parser.add_argument('-t', '--tautomers', help='Generate all tautomers', action='store_true')
    args = parser.parse_args()

    if args.mol is not None:
        m = Chem.MolFromMolFile(args.mol)
        outname = os.path.splitext(args.mol)[0] + ".sdf"
    if args.smiles is not None:
        m = Chem.MolFromSmiles(args.smiles)
        outname = args.smiles + ".sdf"
    if args.tautomers:
        enumerator = rdMolStandardize.TautomerEnumerator()
        tauts = enumerator.Enumerate(m)
        w = Chem.SDWriter(outname)
        for i, taut in enumerate(tauts):
            taut = fixmol(taut)
            w.write(taut)
        sys.exit(0)
    m = fixmol(m)
    w = Chem.SDWriter(outname)
    w.write(m)