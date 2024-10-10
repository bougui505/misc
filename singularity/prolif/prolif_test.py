#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Oct 10 11:27:13 2024
# see: https://prolif.readthedocs.io/en/stable/notebooks/docking.html#docking

import MDAnalysis as mda
import prolif as plf
from rdkit import Chem

# protein_file = "../Receptors/urok.mol2"
# u = mda.Universe(protein_file)
# u.atoms.guess_bonds()
# protein_mol = plf.Molecule.from_mda(u)
protein_file = "receptor.pdb" # "../mkDUDE/all/urok/receptor.pdb"
poses_path = "../Poses/fnta_decoys/fnta_Ligands.sdf"  # "../Poses/urok_actives/urok_Ligands.sdf"

rdkit_prot = Chem.MolFromPDBFile(protein_file, removeHs=True)
protein_mol = plf.Molecule(rdkit_prot)

pose_iterable = plf.sdf_supplier(poses_path)
# use default interactions
fp = plf.Fingerprint()
# run on your poses
fp.run_from_iterable(pose_iterable, protein_mol)
fp.to_pickle("fingerprint.pkl")
fp = plf.Fingerprint.from_pickle("fingerprint.pkl")
ligand, resid = list(fp.ifp[0].keys())[0]
available = plf.Fingerprint.list_available()
print(f"{available=}")
print(f"{dir(ligand)=}")
print(f"{dir(resid)=}")
print("--")
for pose_index in fp.ifp:
    for k in fp.ifp[pose_index]:
        ligand = k[0]
        resid = k[1]
        for interaction in fp.ifp[pose_index][k]:
            print(f"{pose_index=}")
            print(f"{ligand.name=}")
            print(f"{ligand.chain=}")
            print(f"{ligand.number=}")
            print(f"{resid.name=}")
            print(f"{resid.chain=}")
            print(f"{resid.number=}")
            print(f"{interaction=}")
            distance = fp.ifp[pose_index][k][interaction][0]["distance"]
            print(f"{distance=}")
            print("--")
