#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

from misc.sequences.sequence_identity import get_sequence
import pickle
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import os

DIRPATH = os.path.realpath(__file__)
DIRPATH = os.path.dirname(DIRPATH)
INDEXPATH = f"{DIRPATH}/index.pkl"
INDEX = pickle.load(open(INDEXPATH, "rb"))


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def align(pdb1, pdb2, index=None):
    if index is None:
        index = INDEX
    pdbcode1 = pdb1[:4]
    chain1 = pdb1[-1]
    pdbcode2 = pdb2[:4]
    chain2 = pdb2[-1]
    seq1 = get_sequence.get_sequence(pdbcode=pdbcode1, index=index, chain=chain1)
    seq2 = get_sequence.get_sequence(pdbcode=pdbcode2, index=index, chain=chain2)
    alignments = pairwise2.align.globalxx(seq1, seq2)
    alignment = alignments[0]
    # den = alignment.end
    den = min(len(seq1), len(seq2))
    # den = len(seq1)
    sequence_identity = float(alignment[2]) / float(den)
    return alignment, sequence_identity


if __name__ == "__main__":
    import sys
    import doctest
    import argparse

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("--index")
    parser.add_argument(
        "--pdb1", help="Give the pdb code and the chain as PDB_CHAIN. E.g. 4ci0_A"
    )
    parser.add_argument(
        "--pdb2", help="Give the pdb code and the chain as PDB_CHAIN. E.g. 4ci0_A"
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    args = parser.parse_args()

    if args.test:
        doctest.testmod(
            optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
        )
        sys.exit()

    if args.index is not None:
        index = pickle.load(open(args.index, "rb"))
    else:
        index = None
    alignment, seq_identity = align(args.pdb1, args.pdb2, index)
    print(format_alignment(*alignment))
    print(f"seq_identity: {seq_identity:.4f}")
