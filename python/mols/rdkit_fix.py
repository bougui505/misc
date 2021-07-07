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
    parser.add_argument('-m', '--mol', help='input mol file to fix. Multiple mol files can be given.', nargs='+')
    parser.add_argument('-s', '--smiles', help='input smiles file to convert in 3D and fix. Multiple smiles can be given.', nargs='+')
    parser.add_argument('-t', '--tautomers', help='generate all tautomers', action='store_true')
    args = parser.parse_args()

    mlist = []
    outnames = []
    if args.mol is not None:
        for mol in args.mol:
            m = Chem.MolFromMolFile(mol)
            outname = os.path.splitext(mol)[0] + ".sdf"
            mlist.append(m)
            outnames.append(outname)
    if args.smiles is not None:
        for smiles in args.smiles:
            m = Chem.MolFromSmiles(smiles)
            outname = smiles + ".sdf"
            mlist.append(m)
            outnames.append(outname)
    if args.tautomers:
        for m, outname in zip(mlist, outnames):
            enumerator = rdMolStandardize.TautomerEnumerator()
            tauts = enumerator.Enumerate(m)
            w = Chem.SDWriter(outname)
            for i, taut in enumerate(tauts):
                taut = fixmol(taut)
                w.write(taut)
        sys.exit(0)
    for m, outname in zip(mlist, outnames):
        m = fixmol(m)
        w = Chem.SDWriter(outname)
        w.write(m)
