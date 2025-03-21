#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-12-18 11:15:44 (UTC+0100)

import os
import re
import sys

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize


def molfromsmiles(smiles):
    m = Chem.MolFromSmiles(smiles)
    m = fixmol(m)
    return m


def coordsfromsmiles(smiles):
    m = molfromsmiles(smiles)
    coords = get_coords(m)
    return coords


def fixmol(mol, constrain=None, template=None):
    if template is not None:
        # print(f"Assign Bonds from {template}")
        mol = AllChem.AssignBondOrdersFromTemplate(template, mol)
        # mol = Chem.AddHs(mol)
        # AllChem.EmbedMolecule(mol)
        # AllChem.MMFFOptimizeMolecule(mol)
    if constrain is not None:
        # print(f"Constrain conformer from {mol}")
        mol = AllChem.ConstrainedEmbed(mol, constrain)
    if all(elem is None for elem in [constrain, template]):
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
    return mol


def get_coords(mol):
    c = mol.GetConformers()[0]
    coords = c.GetPositions()
    return coords


if __name__ == '__main__':
    import argparse

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-m', '--mol', help='input mol file to fix. Multiple mol files can be given.', nargs='+')
    parser.add_argument('--constrain', help='Constrained the molecule to get this conformation')
    parser.add_argument(
        '--template',
        help=
        'Template smiles file to assign bond orders to the molecule (--mol) based on the bond orders in this template molecule'
    )
    parser.add_argument('-s',
                        '--smiles',
                        help='input smiles file to convert in 3D and fix. Multiple smiles can be given.',
                        nargs='+')
    parser.add_argument('-o', '--out', help='Optional output name')
    parser.add_argument('-t', '--tautomers', help='generate all tautomers', action='store_true')
    args = parser.parse_args()

    if args.constrain is not None:
        constrain = AllChem.MolFromMolFile(args.constrain)
    else:
        constrain = None
    if args.template is not None:
        template = AllChem.MolFromSmiles(args.template)
    else:
        template = None
    mlist = []
    outnames = []
    if args.mol is not None:
        for mol in args.mol:
            m = Chem.MolFromMolFile(mol)
            outname = os.path.splitext(mol)[0] + ".sdf"
            mlist.append(m)
            outnames.append(outname)
    n = len(mlist)
    if args.smiles is not None:
        for smiles in args.smiles:
            m = Chem.MolFromSmiles(smiles)
            outname = re.sub('[^A-Za-z0-9]+', '', smiles)
            outname += ".sdf"
            mlist.append(m)
            outnames.append(outname)
    if args.tautomers:
        for i, (m, outname) in enumerate(zip(mlist, outnames)):
            sys.stdout.write(f'Fixing mol {i+1}/{n}\r')
            sys.stdout.flush()
            enumerator = rdMolStandardize.TautomerEnumerator()
            tauts = enumerator.Enumerate(m)
            w = Chem.SDWriter(outname)
            for i, taut in enumerate(tauts):
                taut = fixmol(taut, constrain=constrain, template=template)
                w.write(taut)
        print()
        sys.exit(0)
    for i, (m, outname) in enumerate(zip(mlist, outnames)):
        # sys.stdout.write(f'Fixing mol {i+1}/{n}\r')
        # sys.stdout.flush()
        m = fixmol(m, constrain=constrain, template=template)
        if args.out is not None:
            outname = args.out
        w = Chem.SDWriter(outname)
        w.write(m)
    print()
