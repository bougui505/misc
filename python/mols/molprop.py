#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri May  9 11:14:02 2025

import os
import sys

import typer
from rdkit import Chem
from rdkit.Chem import QED, RDConfig, SaltRemover, rdFMCS, rdRascalMCES
from rdkit.Chem.Scaffolds import MurckoScaffold

# See: https://mattermodeling.stackexchange.com/a/8544
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
# now you can import sascore!
import sascorer

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.callback()
def callback(debug:bool=False):
    """
    Compute various molecular properties from the SMILES file given in the
    standard input.
    """
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = debug

@app.command()
def qed():
    """
    Compute the QED (Quantitative Estimate of Drug-likeness) from the
    SMILES file given in the standard input.
    """
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # Compute QED using rdkit
            qed = QED.qed(mol)
            print(f"{line} qed: {qed:.3f}")
        except:
            continue

@app.command()
def sascore():
    """
    Compute the SAScore (Synthetic Accessibility Score) from the
    SMILES file given in the standard input.
    Characterize molecule synthetic accessibility as a score between 1 (easy to make) and 10 (very difficult to make).
    See: https://github.com/rdkit/rdkit/tree/master/Contrib/SA_Score
    See: https://mattermodeling.stackexchange.com/a/8544
    See: https://jcheminf.biomedcentral.com/articles/10.1186/1758-2946-1-8
    """
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # Compute SAScore using sascorer
            sascore = sascorer.calculateScore(mol)
            print(f"{line} sascore: {sascore:.3f}")
        except:
            continue

@app.command()
def n_heavy():
    """
    Count the number of heavy atoms in the molecule from the SMILES
    file given in the standard input.
    """
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # Count the number of heavy atoms
            n_heavy = mol.GetNumHeavyAtoms()
            print(f"{line} n_heavy: {n_heavy}")
        except:
            continue

@app.command()
def fp_sim(ref:str):
    """
    Compute the Tanimoto similarity between the fingerprints of the
    molecules from the SMILES file given in the standard input and a 
    reference molecule.\n
    """
    ref_mol = Chem.MolFromSmiles(ref)
    if ref_mol is None:
        print(f"Error: {ref} is not a valid SMILES", file=sys.stderr)
        exit(1)
    try:
        Chem.SanitizeMol(ref_mol)
        ref_fp = Chem.RDKFingerprint(ref_mol)
    except:
        print(f"Error: {ref} is not a valid SMILES", file=sys.stderr)
        exit(1)

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # Compute the Tanimoto similarity
            fp = Chem.RDKFingerprint(mol)
            sim = Chem.DataStructs.TanimotoSimilarity(ref_fp, fp)
            print(f"{line} sim({ref}): {sim:.3f}")
        except:
            continue

@app.command()
def murcko_sim(ref:str):
    """
    Compute the Tanimoto similarity between the Murcko fingerprints of the
    molecules from the SMILES file given in the standard input and a 
    reference molecule.\n
    """
    ref_mol = Chem.MolFromSmiles(ref)
    if ref_mol is None:
        print(f"Error: {ref} is not a valid SMILES", file=sys.stderr)
        exit(1)
    try:
        Chem.SanitizeMol(ref_mol)
        ref_core = MurckoScaffold.GetScaffoldForMol(ref_mol)
        ref_fp = Chem.RDKFingerprint(ref_core)
    except:
        print(f"Error: {ref} is not a valid SMILES", file=sys.stderr)
        exit(1)

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # Compute the Tanimoto similarity
            core = MurckoScaffold.GetScaffoldForMol(mol)
            fp = Chem.RDKFingerprint(core)
            sim = Chem.DataStructs.TanimotoSimilarity(ref_fp, fp)
            print(f"{line} murcko_sim({ref}): {sim:.3f}")
        except:
            continue

@app.command()
def rascal_sim(ref:str):
    """
    Compute the Tanimoto similarity between the RASCAL fingerprints of the
    molecules from the SMILES file given in the standard input and a 
    reference molecule.\n
    """
    opts = rdRascalMCES.RascalOptions()
    opts.singleLargestFrag = True
    opts.allBestMCESs = True
    opts.ignoreAtomAromaticity = True
    opts.ringMatchesRingOnly = False
    opts.completeAromaticRings = False
    ref_mol = Chem.MolFromSmiles(ref)
    if ref_mol is None:
        print(f"Error: {ref} is not a valid SMILES", file=sys.stderr)
        exit(1)
    try:
        Chem.SanitizeMol(ref_mol)
    except:
        print(f"Error: {ref} is not a valid SMILES", file=sys.stderr)
        exit(1)

    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            rascal_matches = rdRascalMCES.FindMCES(ref_mol, mol, opts)
            sim = rascal_matches[0].similarity
            print(f"{line} rascal_sim({ref}): {sim:.3f}")
        except:
            continue

@app.command()
def smiles(infmt:str="sdf"):
    """
    Convert the stdin to SMILES format.
    """
    sdfblock = ""
    for i, line in enumerate(sys.stdin):
        if infmt == "sdf":
            sdfblock += line
            if line.startswith("$$$$"):
                # Convert the SDF block to SMILES
                suppl = Chem.SDMolSupplier()
                suppl.SetData(sdfblock)
                for mol in suppl:
                    if mol is None:
                        continue
                    try:
                        Chem.SanitizeMol(mol)
                        smiles = Chem.MolToSmiles(mol)
                        # recover the sdf fields
                        for field in mol.GetPropNames():
                            if field == "SMILES":
                                continue
                            try:
                                value = mol.GetProp(field)
                                smiles += f" {field.replace(' ', '_')}: {value}"
                            except:
                                pass
                        print(smiles)
                    except:
                        continue
                sdfblock = ""
        else:
            print(f"{infmt} format not supported")
            exit(1)

@app.command()
def stripmol():
    """
    Strip the molecule from the SMILES file given in the standard input.\n
    Remove salts from the molecule and keep only the main molecule.
    """
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
            # Strip the molecule
            remover = SaltRemover.SaltRemover()
            mol = remover.StripMol(mol)
            smiles = Chem.MolToSmiles(mol)
            print(f"{line} stripped: {smiles}")
        except:
            continue

@app.command()
def rec():
    """
    Convert the SMILES file given in the standard input to a recfile format.\n
    \n
    input format:\n
    \n
    SMILES1 field11: value11 field12: value12 ...\n
    SMILES2 field21: value21 field22: value22 ...\n
    \n
    output format:\n
    \n
    smiles=SMILES1\n
    field1=value1\n
    field2=value2\n
    ...\n
    --\n
    smiles=SMILES2\n
    field1=value1\n
    field2=value2\n
    ...\n
    --\n
    """
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        print(f"smiles={smiles}")
        other = line.split()[1:]
        fields = other[0::2]
        values = other[1::2]
        for field, value in zip(fields, values):
            field = field.replace(":", "")
            print(f"{field}={value}")
        print("--")


if __name__ == "__main__":
    # import doctest

    # @app.command()
    # def test():
    #     """
    #     Test the code
    #     """
    #     doctest.testmod(
    #         optionflags=doctest.ELLIPSIS \
    #                     | doctest.REPORT_ONLY_FIRST_FAILURE \
    #                     | doctest.REPORT_NDIFF
    #     )

    # @app.command()
    # def test_func(func:str):
    #     """
    #     Test the given function
    #     """
    #     print(f"Testing {func}")
    #     f = getattr(sys.modules[__name__], func)
    #     doctest.run_docstring_examples(
    #         f,
    #         globals(),
    #         optionflags=doctest.ELLIPSIS \
    #                     | doctest.REPORT_ONLY_FIRST_FAILURE \
    #                     | doctest.REPORT_NDIFF,
    #     )

    app()
