#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jun 11 14:18:45 2024

import os
import tempfile
from os.path import isfile

import numpy as np
import pymol2
import scipy.spatial.distance as scidist
from misc.protein.coords_loader import get_coords  # type: ignore
from rdkit import Chem
from scipy.optimize import linear_sum_assignment


def get_coords(mol):
    coords = []
    for i, atom in enumerate(mol.GetAtoms()):
        positions = mol.GetConformer().GetAtomPosition(i)
        # print(atom.GetSymbol(), positions.x, positions.y, positions.z)
        coords.append([positions.x, positions.y, positions.z])
    coords = np.asarray(coords)
    return coords

def get_rmsd(mol, molref, coord_ref=None):
    coords = get_coords(mol)
    if coord_ref is None:
        coord_ref = get_coords(molref)
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


def load_or_fetch(pdb, name):
    if os.path.isfile(pdb):
        print(f"loading={pdb}")
        pymol.cmd.load(pdb, name)
    else:
        print(f"fetching={pdb}")
        pymol.cmd.fetch(pdb, name)

def hungarian(coords1, coords2):
    """"""
    pdist = scidist.cdist(coords1, coords2)
    row_ind, col_ind = linear_sum_assignment(pdist)
    rmsd = np.sqrt(((coords1[row_ind, :] - coords2[col_ind, :])**2).mean(axis=0).sum())
    return rmsd

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--mol2ref", help="Reference mol2 file")
    parser.add_argument("--sdf", help="sdf file with molecules to compare with reference")
    parser.add_argument("--pdbref", help="Structure with molecule of reference")
    parser.add_argument("--pdb", help="Structure with molecule to compare with reference")
    parser.add_argument("--selref", help="Selection for ligand of reference (see --molref)")
    parser.add_argument("--sel", help="Selection for ligand to compare with reference (see --mol)")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.mol2ref is not None:
        molref = Chem.MolFromMol2File(args.mol2ref, sanitize=True)
    else:
        molref = None
    if args.sdf is not None and molref is not None:
        suppl = Chem.SDMolSupplier(args.sdf, sanitize=True)

        coord_ref = get_coords(molref)

        for mol in suppl:
            rmsd, match_ratio, molname = get_rmsd(mol, molref, coord_ref)
            print(f"{molname=}")
            print(f"{match_ratio=:.3g}")
            print(f"{rmsd=:.4g}")
            print("--")
    if args.pdbref is not None and args.pdb is not None:
        with pymol2.PyMOL() as pymol:
            # tmpdir = tempfile.mkdtemp()
            # print(f"{tmpdir=}")
            fetch_path=os.path.expanduser("~/pdb")
            pymol.cmd.set("fetch_path", fetch_path)
            pymol.cmd.set("fetch_type_default", "mmtf")
            load_or_fetch(args.pdbref, "ref")
            load_or_fetch(args.pdb, "other")
            out = pymol.cmd.align("other", "ref")
            rmsd_all = out[0]
            print(f"{rmsd_all=}")
            coords_ref = pymol.cmd.get_coords(args.selref)
            coords = pymol.cmd.get_coords(args.sel)
            print(f"{coords_ref.shape=}")
            print(f"{coords.shape=}")
            rmsd_lig = hungarian(coords, coords_ref)
            print(f"{rmsd_lig=}")
            # pymol.cmd.h_add()
            # pymol.cmd.save(f"{tmpdir}/ref.mol2", selection=args.selref)
            # pymol.cmd.save(f"{tmpdir}/other.mol2", selection=args.sel)
            # molref = Chem.MolFromMol2File(f"{tmpdir}/ref.mol2", sanitize=False)
            # mol = Chem.MolFromMol2File(f"{tmpdir}/other.mol2", sanitize=False)
            # smi____ = Chem.MolToSmiles(mol)
            # smi_ref = Chem.MolToSmiles(molref)
            # print(f"{smi____=}")
            # print(f"{smi_ref=}")
            # rmsd_lig, match_ratio, molname = get_rmsd(mol, molref)
            # print(f"{rmsd_lig=}")
