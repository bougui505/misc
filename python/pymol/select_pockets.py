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
import os
from pymol import cmd
import numpy as np
from tqdm import tqdm


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def get_coms(sdflist):
    """
    """
    comlist = []
    for i, sdf in enumerate(sdflist):
        cmd.load(sdf, f'l{i}')
        coords = cmd.get_coords(f'l{i}')
        cmd.delete(f'l{i}')
        com = coords.mean(axis=0)
        comlist.append(com)
    return np.asarray(comlist)


def get_pockets(pdb, sdflist, radius, natom_cutoff=None, smiles=None):
    """
    Split pdb to get natom_cutoff maximum number of atoms
    """
    comlist = get_coms(sdflist)
    selection = f'byres((com around {radius}) and pdb)'
    cmd.load(pdb, 'pdb')
    index = 0
    current_smiles = []
    current_sdf = []
    for i, com in enumerate(comlist):
        current_sdf.append(sdflist[i])
        if smiles is not None:
            current_smiles.append(smiles[i])
        cmd.pseudoatom(object='com', pos=tuple(com))
        natoms = cmd.select(name='pockets', selection=selection)
        if natom_cutoff is not None:
            # print(f'npockets: {i+1}|natoms: {natoms}')
            if natoms > natom_cutoff:
                outfilename = save_pocket(pdb, selection, index=index)
                if smiles is not None:
                    # print(smiles[i], outfilename, sdflist[i])
                    for sm, sd in zip(current_smiles, current_sdf):
                        print(sm, outfilename, sd)
                index += 1
                current_smiles = []
                current_sdf = []
    try:
        natoms = cmd.select(name='pockets', selection=selection)
        outfilename = save_pocket(pdb, selection, index=index)
        if smiles is not None:
            # print(smiles[i], outfilename, sdflist[i])
            for sm, sd in zip(current_smiles, current_sdf):
                print(sm, outfilename, sd)
    except:
        pass
    cmd.delete('pdb')


def save_pocket(pdb, selection, index=None):
    cmd.select(name='pockets', selection=selection)
    cmd.delete('com')
    if index is None:
        suffix = '_pocket.pdb'
    else:
        suffix = f'_pocket_{index}.pdb'
    outfilename = os.path.splitext(pdb)[0] + suffix
    cmd.save(outfilename, selection='pockets')
    cmd.delete('pocket')
    return outfilename


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
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
    parser.add_argument('--pdb')
    parser.add_argument('--sdf', nargs='+')
    parser.add_argument(
        '--smi',
        help=
        'SMILES to read data. Format must be 3 columns: smiles rec lig. rec and lig are the path to receptor file and ligand file respectively'
    )
    parser.add_argument('--radius', type=float, required=True)
    parser.add_argument('--natom_cutoff', type=int, help='Split the output pdb if more than natom_cutoff atoms')
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

    if args.pdb is not None and args.sdf is not None:
        get_pockets(args.pdb, args.sdf, args.radius, natom_cutoff=args.natom_cutoff)
    if args.smi is not None:
        n = sum(1 for i in open(args.smi))
        recset = set()
        ligs = []
        smiles = []
        bn, ext = os.path.splitext(args.smi)
        outsmifilename = bn + "_pocket" + ext
        for i, line in tqdm(enumerate(open(args.smi)), total=n):
            line = line.strip()
            if line.startswith('#'):
                continue
            if i > 0:
                rec = _rec_
            _smiles_, _rec_, _lig_ = line.split()
            if i == 0:
                rec = _rec_
            if _rec_ not in recset:
                if len(recset) > 0:
                    # print(rec, ligs, smiles)
                    get_pockets(rec, ligs, args.radius, natom_cutoff=args.natom_cutoff, smiles=smiles)
                ligs = []
                smiles = []
                recset.add(_rec_)
            ligs.append(_lig_)
            smiles.append(_smiles_)
        # print(rec, ligs, smiles)
        get_pockets(rec, ligs, args.radius, natom_cutoff=args.natom_cutoff, smiles=smiles)
