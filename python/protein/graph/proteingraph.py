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
import pymol2
import numpy as np
import torch
import scipy.spatial.distance as scidist
import logging
if not os.path.isdir('logs'):
    os.mkdir('logs')
logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
logging.info(f"################ Starting {__file__} ################")

AMINO_ACIDS = [
    'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER',
    'THR', 'TRP', 'TYR', 'VAL', 'XXX'
]

ATOMS = [
    'CE3', 'CG1', 'ND1', 'N', 'NE1', 'CG2', 'ND2', 'CZ', 'O', 'CH2', 'NE2', 'C', 'CA', 'CB', 'NH1', 'NE', 'OD1', 'NH2',
    'CD', 'CE', 'OE1', 'OD2', 'CZ2', 'OG', 'OE2', 'OH', 'CZ3', 'OG1', 'SD', 'CG', 'CD1', 'CE1', 'CD2', 'SG', 'CE2',
    'NZ', 'XXX'
]


def getclasslist(mapping, inplist):
    classlist = []
    for e in inplist:
        if e in mapping:
            classlist.append(mapping[e])
        else:
            classlist.append(mapping['XXX'])
            log(f"unknown key {e}")
    return torch.tensor(classlist)


def seq_to_1hot(seqlist):
    """
    >>> seqlist = ['GLY', 'SER', 'GLN', 'ILE', 'PRO', 'ALA', 'SER', 'GLU', 'GLN', 'GLU', 'DLY', 'THR', 'LEU']
    >>> len(seqlist)
    13
    >>> onehot = seq_to_1hot(seqlist)
    >>> onehot.shape
    torch.Size([13, 21])
    >>> onehot[0]
    tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> onehot[10]
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    """
    mapping = dict(zip(AMINO_ACIDS, range(len(AMINO_ACIDS))))
    classlist = getclasslist(mapping, seqlist)
    onehot = torch.nn.functional.one_hot(classlist, num_classes=len(AMINO_ACIDS))
    return onehot


def atomlist_to_1hot(atomlist):
    """
    >>> atomlist = ['ND1', 'CE3', 'CG1', 'Y', 'CA']
    >>> len(atomlist)
    5
    >>> onehot = atomlist_to_1hot(atomlist)
    >>> onehot.shape
    torch.Size([5, 37])
    >>> onehot[0]
    tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    >>> onehot[3]
    tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
    """
    mapping = dict(zip(ATOMS, range(len(ATOMS))))
    classlist = getclasslist(mapping, atomlist)
    onehot = torch.nn.functional.one_hot(classlist, num_classes=len(ATOMS))
    return onehot


def prot_to_graph(pdb, selection=None):
    """
    >>> node_features, edge_index, edge_features = prot_to_graph('1t4e.pdb')
    >>> node_features.shape
    torch.Size([1568, 58])
    >>> edge_index.shape
    torch.Size([2, 123218])
    >>> edge_features.shape
    torch.Size([123218])
    """
    log(f"pdbfile: {pdb}")
    if selection is None:
        selection = 'polymer.protein'
    with pymol2.PyMOL() as p:
        p.cmd.load(pdb, 'prot')
        coords = p.cmd.get_coords(selection=selection)
        space = {'resnames': [], 'atomnames': []}
        p.cmd.iterate(selection=selection, space=space, expression='resnames.append(resn); atomnames.append(name)')
        resnames = np.asarray(space['resnames'])
        atomnames = np.asarray(space['atomnames'])
    resnames_onehot = seq_to_1hot(resnames)
    atomnames_onehot = atomlist_to_1hot(atomnames)
    node_features = torch.cat((resnames_onehot, atomnames_onehot), dim=1)
    dmat = scidist.squareform(scidist.pdist(coords))
    edge_index = torch.tensor(np.asarray(np.where(dmat < 8.)))  # edge_index has shape [2, E] with E the number of edges
    edge_features = torch.tensor(dmat[tuple(edge_index)])
    return node_features, edge_index, edge_features


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
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
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
    if args.pdb is not None:
        prot_to_graph(args.pdb)
