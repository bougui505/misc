#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jan 27 09:35:49 2026

from io import StringIO

import requests
import typer
from Bio import SeqIO

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

def main(
        acc: str = typer.Argument(..., help="UniProt accession number"),
    ):
    # Get features to find the "Chain"
    gff = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.gff").text
    for line in gff.splitlines():
        if "Chain" in line:
            parts = line.split("\t")
            start, end = int(parts[3]), int(parts[4])
            
            # Get sequence and slice
            fasta = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta").text
            record = SeqIO.read(StringIO(fasta), "fasta")
            print(f">Mature_{acc}\n{record.seq[start-1:end]}")
            break

if __name__ == "__main__":
    import doctest
    import sys
    app()
