#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Aug 27 14:14:23 2025

import typer
from RAscore import RAscore_XGB #For XGB based models

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.command()
def rascore():
    xgb_scorer = RAscore_XGB.RAScorerXGB()
    for i, line in enumerate(sys.stdin):
        line = line.strip()
        smiles = line.split()[0]
        try:
            rascore = xgb_scorer.predict(smiles)
            print(f"{line} rascore: {rascore:.3f}")
        except:
            continue

if __name__ == "__main__":
    import doctest
    import sys

    app()

