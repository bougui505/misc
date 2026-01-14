#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Jan 14 09:39:24 2025

import MDAnalysis as mda
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
def load_trajectory(traj_file: str, top_file: str = None):
    """
    Load an MD trajectory using MDAnalysis
    
    Args:
        traj_file (str): Path to the trajectory file
        top_file (str, optional): Path to the topology file. If None, will try to infer from traj_file
    """
    try:
        if top_file is None:
            # Try to infer topology from trajectory file
            u = mda.Universe(traj_file)
        else:
            u = mda.Universe(top_file, traj_file)
        
        print(f"Loaded trajectory with {u.trajectory.n_frames} frames")
        print(f"System has {len(u.atoms)} atoms")
        print(f"Topology: {u.atoms[0].resname} {u.atoms[0].resid}")
        
        return u
    except Exception as e:
        print(f"Error loading trajectory: {e}")
        raise

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
