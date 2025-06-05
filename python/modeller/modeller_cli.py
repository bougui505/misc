#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jun  5 09:56:16 2025

import os

import modeller
import modeller.automodel as automodel
import typer

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

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
def get_pir(pdb:str) -> str:
    """
    Get the PIR sequence file from a PDB file.

    :param pdb: Path to the PDB file.
    
    """
    env = modeller.environ()
    mdl = modeller.model(env, file=pdb)
    aln = modeller.alignment(env)
    aln.append_model(mdl, align_codes=pdb, atom_files=pdb)
    outfasta = f'{os.path.splitext(pdb)[0]}.pir'
    aln.write(file=outfasta)

@app.command()
def mutate(
    pdb:str,
    pirfile:str,
    mutant_name:str,
    ):
    """
    Mutate a residue in a PDB file using Modeller.\n
    :param pdb: Path to the PDB file.\n
    :param pirfile: Path to the PIR file.\n
    :param mutant_name: Name of the mutant in the PIR file (e.g. "T26A").\n
                        Multiple mutants can be specified by separating them with commas (e.g. "T26A,T27G").\n
    \n
    Example PIR file content:\n
    >P1;1ycr.pdb\n
    structureX:1ycr.pdb:25:A:+98:B:::-1.00:-1.00\n
    ETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKI\n
    YTMIYRNLVV/ETFSDLWKLLPEN*\n
    >P1;T26A\n
    sequence:T26A:25:A:+98:B:::-1.00:-1.00\n
    EALVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKI\n
    YTMIYRNLVV/ETFSDLWKLLPEN*\n
    """
    env = modeller.environ()
    mdl = modeller.model(env, file=pdb)
    aln = modeller.alignment(env)
    aln.append_model(mdl, align_codes=pdb, atom_files=pdb)
    mutant_names = mutant_name.split(',')
    for mutant in mutant_names:
        a = automodel.automodel(env, alnfile=pirfile, knowns=pdb, sequence=mutant)
        a.starting_model = 1
        a.ending_model = 1
        #  set the numbering of the residues in the output model according to the PIR file
        a.make()

if __name__ == "__main__":
    import doctest
    import sys

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
