#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jan 27 09:35:49 2026

from io import StringIO
import argparse
import sys

import requests
from Bio import SeqIO

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

def main(acc):
    """
    Retrieve a fasta sequence of the mature protein (without signal peptide) from a uniprot id
    """
    # Try to get features to find the "Chain"
    try:
        gff = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.gff").text
        for line in gff.splitlines():
            if "Chain" in line:
                parts = line.split("\t")
                start, end = int(parts[3]), int(parts[4])
                
                # Get sequence and slice
                fasta = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta").text
                record = SeqIO.read(StringIO(fasta), "fasta")
                print(f">Mature_{acc}  {start}-{end}\n{record.seq[start-1:end]}")
                break
        else:
            # If no chain found, get the full sequence
            fasta = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta").text
            try:
                record = SeqIO.read(StringIO(fasta), "fasta")
            except ValueError:
                sys.exit(0)
            print(f">Full_{acc}\n{record.seq}")
    except requests.exceptions.RequestException:
        # If GFF request fails, get the full sequence
        fasta = requests.get(f"https://rest.uniprot.org/uniprotkb/{acc}.fasta").text
        record = SeqIO.read(StringIO(fasta), "fasta")
        print(f">Full_{acc}\n{record.seq}")

if __name__ == "__main__":
    import doctest
    parser = argparse.ArgumentParser(description="Retrieve a fasta sequence of the mature protein (without signal peptide) from a uniprot id")
    parser.add_argument("acc", help="UniProt accession number. The script will retrieve the mature sequence (without signal peptide)")
    args = parser.parse_args()
    main(args.acc)
