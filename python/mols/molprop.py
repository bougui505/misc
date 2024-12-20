#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
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
# Compute the QED (see: https://www.nature.com/articles/nchem.1243)

import gzip
import os

from rdkit import Chem
from rdkit.Chem import QED
from tqdm import tqdm

AVAILABLE_PROPERTIES = ['qed', 'num_heavy']


def get_props(smifilename, properties):
    """
    properties: list of properties to compute
    """
    unavailable_properties = set(properties) - set(AVAILABLE_PROPERTIES)
    assert len(unavailable_properties) == 0, f"Unavailable properties: {unavailable_properties}"
    outfilename = os.path.splitext(smifilename.replace(".gz", ""))[0] + "_properties.smi.gz"
    num_lines = sum(1 for line in gzip.open(smifilename))
    with gzip.open(smifilename, 'rt') as smifile:
        with gzip.open(outfilename, 'wt') as outfile:
            for line in tqdm(smifile, total=num_lines):
                line = line.strip()
                if line.startswith("#"):  # type: ignore
                    header = line
                    for prop in properties:
                        header += f" #{prop}"  # type: ignore
                    outfile.write(header + "\n")  # type: ignore
                    continue
                fields = line.split()
                smiles = fields[0]
                others = fields[1:]
                outstr = f"{smiles} {' '.join(others)}"  # type: ignore
                mol = Chem.MolFromSmiles(smiles)  # type: ignore
                for prop in properties:
                    if prop == 'qed':
                        qed, outstr = get_qed(mol, outstr)
                    if prop == 'num_heavy':
                        num_heavy, outstr = get_num_heavy(mol, outstr)
                outfile.write(outstr + "\n")  # type: ignore


def get_qed(mol, outstr):
    if mol is not None:
        qed = QED.qed(mol)
    else:
        qed = -1
    outstr += f" {qed:.2f}"
    return qed, outstr


def get_num_heavy(mol, outstr):
    if mol is not None:
        num_heavy = mol.GetNumHeavyAtoms()
    else:
        num_heavy = -1
    outstr += f" {num_heavy}"
    return num_heavy, outstr


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == '__main__':
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--smi', help="Gzipped SMILES file to process")
    parser.add_argument('--properties',
                        help=f"Properties to compute. Available properties: {AVAILABLE_PROPERTIES}",
                        nargs='+')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.smi is not None:
        get_props(args.smi, properties=args.properties)
