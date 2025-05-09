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
from rdkit.Chem import QED, RDConfig

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
    This is a template file for a Python script using Typer.
    It contains a main function and a test function.
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

if __name__ == "__main__":
    import doctest

    @app.command()
    def test():
        """
        Test the code
        """
        doctest.testmod(
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF
        )

    @app.command()
    def test_func(func:str):
        """
        Test the given function
        """
        print(f"Testing {func}")
        f = getattr(sys.modules[__name__], func)
        doctest.run_docstring_examples(
            f,
            globals(),
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF,
        )

    app()
