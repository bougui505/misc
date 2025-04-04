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
    ts:bool=True,
    elapsed:bool=True,
    delta:bool=False,
    color:str="green",
):
    """
    Print the time elapsed since the start of the script and the current time
    """
    # read from stdin as a stream
    start = datetime.now()
    last = start
    while True:
        # read a line from stdin
        # line = input()
        line = sys.stdin.readline()
        string, now = format_string(color, ts, elapsed, delta, start, last)
        if line == "":
            break
        print(f"{string}[/{color}]{line}", end="")
        # print(f"[green]123|[/green]{line}", end="")
        last = now
    print(f"{string}[/{color}]##### END OF OUTPUT #####", end="")


def format_string(color, ts, elapsed, delta, start, last):
    string = f"[{color}]"
    now = datetime.now()
    if ts:
        ts_str = now.strftime("%Y%m%d %H:%M:%S")
        string += f"[{color}]{ts_str}"
    if elapsed:
        elapsed_str = now - start
        string += f" t={elapsed_str}"
    if delta:
        delta_str = now - last
        string += f" Δ={delta_str}"
    string += f"|[/{color}]"
    return string, now
        

if __name__ == "__main__":
    import doctest
    import sys

    app()
