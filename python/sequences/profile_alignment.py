#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jan 31 15:16:18 2025

import os
from collections import OrderedDict, defaultdict


def read_fasta(fasta):
    """
    >>> fasta = "data/b2a7_closed_C.fa"
    >>> seqs = read_fasta(fasta)
    """
    sequences = OrderedDict()
    with open(fasta, "r") as f:
        header = None
        seq = None
        for line in f:
            if line.startswith(">"):
                if header is not None:
                    sequences[header] = seq
                header = line.strip()
                seq = ""
            else:
                seq += line.strip()
        sequences[header] = seq
    return sequences

def get_insert_chunk(seq, seqid, chunkdict):
    pos = 0
    for e in seq:
        # print(e, end="")
        if e == "-":
            if pos not in chunkdict:
                chunkdict[pos] = dict()
            if seqid not in chunkdict[pos]:
                chunkdict[pos][seqid] = 0
            chunkdict[pos][seqid]+=1
        else:
            pos += 1
    return chunkdict

def write_aln(fastalist, chunkdict):
    seqlengths = []
    for fasta in fastalist:
        seqlengths.extend([len(seq) for seq in fasta.values()])
    maxseqlen = max(seqlengths)
    for fasta_index, fasta in enumerate(fastalist):
        guide = args.guides[fasta_index]
        seq_guide = list(fasta.values())[guide]
        len_seq_guide = len([e for e in seq_guide if e!="-"])
        for key in fasta:
            print(key)
            seq = fasta[key]
            pos = 0
            done = dict()
            seqlen = 0
            for i, aa in enumerate(seq):
                if pos in chunkdict and pos not in done:
                    if fasta_index in chunkdict[pos]:
                        gap_len = chunkdict[pos][fasta_index]
                    else:
                        gap_len = 0
                    max_gap_len = max(chunkdict[pos].values())
                    toadd = max_gap_len - gap_len
                    print("-"*toadd, end="")
                    seqlen += toadd
                    done[pos] = 1
                print(aa, end="")
                seqlen += 1
                if seq_guide[i] != "-":
                    pos += 1
                    if pos == len_seq_guide:
                        pos = "END"
            print("-"*(maxseqlen - seqlen + 2))
            print("")



if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="Align multiple profiles based on an anchor sequence. E.g ab.fa, cb.fa, db.fa will give an alignment based on sequence b with a, c, d aligned in this frame.")
    parser.add_argument("-f", "--fasta", help="Fasta files to aligned", nargs="+")
    parser.add_argument("-g", "--guides", help="Ids (0-based) of sequences to use as guide for the sequence alignment", type=int, nargs="+")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )
        sys.exit()
    if args.fasta is not None:
        assert args.guides is not None, "Please provide --guides"
        assert len(args.fasta) == len(args.guides), "Please give one guide id per fasta"
        fastalist = []
        for fasta in args.fasta:
            fastalist.append(read_fasta(fasta))  # list of dictionnary of the input fasta
        chunkdict = dict()
        for fasta_index, (guide, fasta) in enumerate(zip(args.guides, fastalist)):
            seq_guide = list(fasta.values())[guide]
            get_insert_chunk(seq_guide, fasta_index, chunkdict)
        write_aln(fastalist, chunkdict)
