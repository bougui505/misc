#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Apr  3 15:22:54 2025

from datetime import datetime

import typer
from rich import print

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.command()
def timer(
    color:str="green",
):
    """
    Print the time elapsed since the start of the script and the current time
    """
    # read from stdin as a stream
    start = datetime.now()
    while True:
        # read a line from stdin
        # line = input()
        line = sys.stdin.readline()
        if line == "":
            break
        now = datetime.now()
        ts = now.strftime("%Y%m%d %H:%M:%S")
        elapsed = now - start
        # elapsed = str(elapsed).split(".")[0]
        print(f"[{color}]{ts} ({elapsed})|[/{color}]{line}", end="")
        

if __name__ == "__main__":
    import doctest
    import sys

    app()
