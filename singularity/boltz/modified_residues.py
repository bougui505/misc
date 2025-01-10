#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# adapted from: https://github.com/benf549/boltz-generalized-covalent-modification/blob/main/boltz_generalized_covalent_inference.ipynb
# creation_date: Fri Jan 10 09:02:26 2025

import argparse
import pickle
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a modified residue for boltz. Adapted from: https://github.com/benf549/boltz-generalized-covalent-modification/blob/main/boltz_generalized_covalent_inference.ipynb")
    parser.add_argument("-s", "--smi", help="SMILES of the modified residue", metavar="O=C(CCSC[C@H](C(O)=O)N)N1CC(C(F)/C(OC)=C(OC)/OC)OC(OC)C1C2=NC=CC2", required=True, type=str)
    parser.add_argument("-a", "--aa", help="Reference residue to modify", metavar="CYS", required=True)
    parser.add_argument("-o", "--out", help="Output CCD code", metavar="CUSTOM", required=True)
    args = parser.parse_args()


    # THIS IS ABSOLUTELY NECESSARY TO MAKE SURE THAT ALL PROPERTIES ARE PICKLED OR ELSE BOLTZ WILL CRASH.
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    ncaa_smiles_str = eval('"' + args.smi.replace('"', '\\"') + '"')  # rf"{args.smi}"
    print(f"{ncaa_smiles_str=}")

    # I'm not sure if this matters as long as it can be substructure matched into the covalent ligand, 
    #   using an alanine might allow this to generalize beyond cysteine modifications and it seems to work in my testing.
    #   You'd probably want to use the largest residue that matches the covalent attachment point to avoid matching to the wrong part of the ligand.
    reference_residue_to_modify = args.aa

    # The new CCD code for the covalent ligand, the first 5 letters are written to the CIF file as the resname.
    output_ccd_code = args.out

    # Load the Boltz Cache Chemical Component Dictionary
    cache_dir = Path("boltz_data")

    ccd_path = cache_dir / 'ccd.pkl'
    with ccd_path.open("rb") as file:
        ccd = pickle.load(file)  # noqa: S301

    smiles_mol = Chem.MolFromSmiles(ncaa_smiles_str)

    # Load Cysteine from the CCD and remove hydrogens
    cys_mol = ccd[reference_residue_to_modify]
    reference_cys_mol = AllChem.RemoveHs(cys_mol)

    # Search for the cysteine substructure in the ncaa molecule
    has_match = smiles_mol.HasSubstructMatch(reference_cys_mol)
    if has_match:
        match_indices = smiles_mol.GetSubstructMatch(reference_cys_mol)
        substruct_to_match = {i.GetProp('name'): match_indices[idx] for idx, i in enumerate(reference_cys_mol.GetAtoms())}

    # Construct mapping of cysteine atom name to atom index in the ncaa molecule
    idx_to_name = {j: i for i, j in substruct_to_match.items()}
    print(f"{idx_to_name=}")

    # Add the metadata boltz expects to the atoms
    for idx, atom in enumerate(smiles_mol.GetAtoms()):
        default_name = f'{atom.GetSymbol()}{str(atom.GetIdx())}'

        # If the index is a canonical cysteine atom, use the cysteine atom name
        name = idx_to_name.get(idx, default_name)

        # Set atom properties
        atom.SetProp('name', name)
        atom.SetProp('alt_name', name) 
        is_leaving = False
        if name == 'OXT':
            is_leaving = True
        atom.SetBoolProp('leaving_atom', is_leaving)

    # Reorder atoms to canonical ordering (N, Ca, C, O, CB, S, ...)
    # Map atom name to atom index in the ncaa molecule 
    curr_atom_order = {atom.GetProp('name'): idx for idx, atom in enumerate(smiles_mol.GetAtoms()) if atom.GetSymbol() != 'H'}

    # Map atom name to atom index in the reference cysteine molecule
    target_atom_order = {}
    for atom in reference_cys_mol.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        target_atom_order[atom.GetProp('name')] = atom.GetIdx()

    # There are atoms not in target_atom_order that are in curr_atom_order so we need to these
    remapped_atom_order = {}
    offset_idx = len(target_atom_order)
    for atom in curr_atom_order:
        if atom in target_atom_order:
            remapped_atom_order[atom] = target_atom_order[atom]
        else:
            remapped_atom_order[atom] = offset_idx
            offset_idx += 1

    # Remove hydrogens and reorder atoms according to the order in the reference cysteine.
    trim = AllChem.RemoveHs(smiles_mol)
    remap_order = {x.GetProp('name'): (remapped_atom_order[x.GetProp('name')], x.GetIdx()) for x in trim.GetAtoms()}
    remap_idx_list = [x[1] for x in sorted(remap_order.values())]
    trim_reordered = Chem.RenumberAtoms(trim, remap_idx_list)

    # Generate a conformer, could make this more sophisticated...
    # Generate a simple conformer, might want to do something more sophisticated or start with a DFT conformer in an SDF file.
    trim_reordered = AllChem.AddHs(trim_reordered)
    AllChem.EmbedMolecule(trim_reordered)
    AllChem.UFFOptimizeMolecule(trim_reordered)

    # Set conformer properties
    for c in trim_reordered.GetConformers():
        c.SetProp('name', 'Ideal')

    # Sanity check renumbering was successful.
    for atom in trim_reordered.GetAtoms():
        if atom.GetSymbol() == 'H':
            continue
        print(atom.GetIdx(), atom.GetPropsAsDict())

    # Save the conformer to the ccd cache and overwrite it.
    ccd[output_ccd_code] = trim_reordered

    ccd_path = cache_dir / 'ccd.pkl'
    with ccd_path.open("wb") as file:
        pickle.dump(ccd, file)

    # To Run Inference:
    # Run boltz predict ./path_to_yaml_file with the following yaml file contents to test the custom modification at residue 2:
    # 
    # version: 1  # Optional, defaults to 1
    # sequences:
    #   - protein:
    #       id: [A]
    #       sequence: AAAAACAAAAAAAAAAA # This doesn't have to have a CYS at position 6 to apply the modification, but it does in this case.
    #       msa: empty
    #       modifications:
    #         - position: 6 
    #           ccd: 'CUSTOM' # Use the custom CCD code we just injected.
