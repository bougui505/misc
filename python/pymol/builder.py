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

from pymol import cmd
import pymol
import chempy


def seq1_to_seq3(seq1):
    """
    Convert array of 1-letter code sequence to 3-letter code.
    seq1 can be an array or a string.
    Return seq3 as array.
    """
    res3 = [
        '---', 'ala', 'asn', 'asp', 'arg', 'cys', 'gln', 'glu', 'gly', 'his', 'ile', 'leu', 'lys', 'met', 'pro', 'phe',
        'ser', 'thr', 'trp', 'tyr', 'val', 'unk', 'ALA', 'ASN', 'ASP', 'ARG', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
        'LEU', 'LYS', 'MET', 'PRO', 'PHE', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'UNK'
    ]
    res1 = '-andrcqeghilkmpfstwyvxANDRCQEGHILKMPFSTWYVX'

    if len(seq1) > 1:
        seq3 = []
        for a1 in seq1:
            try:
                a3 = res3[res1.index(a1)]
                seq3.append(a3)
            except ValueError:
                print("No match for residue: %s" % (a1))
    else:
        seq3 = res3[res1.index(seq1)]

    return seq3


def build_flanking(seq, fragment, anchor, res_anchor):
    """
    See: https://pymolwiki.org/index.php/Modeling_and_Editing_Structures#Adding_and_using_your_own_fragments
    """
    resid = 0
    fragment_path = chempy.fragments.path
    for aa in seq:
        resid += 1
        if resid == 1:
            cmd.editor.build_peptide(aa)
        else:
            if aa != 'X':
                aa = seq1_to_seq3(aa.lower())
                cmd.editor.attach_amino_acid(selection='pk1', amino_acid=aa, center=1)
            else:
                chempy.fragments.path = './'
                cmd.edit(selection1=res_anchor)
                cmd.editor.attach_fragment(selection='pk1', fragment=fragment, anchor=anchor, hydrogen=0)
                chempy.fragments.path = fragment_path
                cmd.edit(f'resid {resid} and name C')
    return resid


if __name__ == '__main__':
    import sys
    import os
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
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-s',
                        '--seq',
                        help='Sequence to build in 1 letter code. Mark the optional input structure as X',
                        required=True)
    parser.add_argument('-i', '--inp', help='Optional structure modified fragment to add on structure')
    parser.add_argument('--anchor', help='atom index of the connecting (hydrogen) atom in the fragment', type=int)
    parser.add_argument('--res_anchor', help='atom selection to connect the fragment on', type=str)
    parser.add_argument('-o', '--out', help='Output file name', required=True)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    if args.inp is not None:
        cmd.load(filename=args.inp, object='X_')
        outfragment = os.path.splitext(os.path.basename(args.inp))[0]
        cmd.save(outfragment + '.pkl', selection='X_')
        cmd.remove('X_')

    resid = build_flanking(args.seq, fragment=outfragment, anchor=args.anchor, res_anchor=args.res_anchor)
    cmd.save(args.out)
