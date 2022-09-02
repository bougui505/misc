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
import os
from pymol import cmd
from misc.randomgen import randomstring
import numpy as np


def get_chain_ids(selection):
    """
    >>> coords, selection = get_coords('1ycr', return_selection=True)
    Fetching 1ycr from the PDB
    >>> coords.shape
    (818, 3)
    >>> selection
    '... and all and present'
    >>> chains = get_chain_ids(selection)
    >>> chains
    ['A', 'A', 'A', 'A', 'A',...
    """
    chains = get_atom_property(selection, prop='chain')
    return chains


def get_atom_property(selection, prop):
    """
    Return a property per atom in selection:

    >>> coords, selection = get_coords('4ci0', selection='name CA and chain B', return_selection=True)
    Fetching 4ci0 from the PDB
    >>> coords.shape
    (228, 3)
    >>> resids = get_atom_property(selection, prop='resi')
    >>> resids
    ['46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '77', '78', '79', '80', '81', '82', '83', '84', '85', '86', '87', '88', '89', '90', '91', '92', '93', '94', '95', '96', '97', '98', '99', '100', '101', '102', '103', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '124', '125', '126', '127', '128', '129', '130', '131', '132', '133', '134', '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '149', '150', '151', '152', '153', '154', '155', '156', '157', '158', '159', '160', '161', '162', '163', '164', '165', '166', '167', '168', '169', '170', '171', '172', '173', '174', '175', '176', '177', '178', '179', '180', '181', '182', '183', '184', '185', '186', '187', '188', '189', '190', '191', '192', '193', '194', '195', '196', '197', '198', '199', '200', '201', '202', '203', '204', '205', '206', '207', '208', '209', '210', '211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222', '223', '224', '225', '226', '227', '228', '229', '230', '231', '232', '233', '234', '235', '236', '237', '238', '239', '240', '241', '242', '243', '244', '245', '246', '247', '248', '249', '250', '251', '252', '253', '254', '255', '256', '257', '258', '259', '260', '261', '262', '263', '264', '265', '266', '267', '268', '269', '270', '271', '272', '273']
    >>> len(resids)
    228
    """
    myspace = {'prop': []}
    cmd.iterate(selection, f'prop.append({prop})', space=myspace)
    return myspace['prop']


def load_pymol_view(view):
    """Read rotation matrix from output of PyMol ``get_view`` command
    Args:
        file (str): Path to file
    """
    view = view.replace('(', '')
    view = view.replace(')', '')
    fields = view.split(',')
    view_matrix = np.asarray(fields, dtype=float)[:16]
    return view_matrix


def get_coords(pdb, selection='all', split_by_chains=False, return_selection=False, view=None, verbose=True):
    """
    >>> coords = get_coords('1ycr')
    Fetching 1ycr from the PDB
    >>> coords.shape
    (818, 3)
    >>> coords = get_coords(os.path.expanduser('~/pdb/1ycr.mmtf'))
    Loading from local file .../pdb/1ycr.mmtf
    >>> coords.shape
    (818, 3)
    >>> coords = get_coords('1ycr', selection='chain A')
    Fetching 1ycr from the PDB
    >>> coords.shape
    (705, 3)
    >>> coords = get_coords('1ycr', split_by_chains=True)
    Fetching 1ycr from the PDB
    >>> [e.shape for e in coords]
    [(705, 3), (113, 3)]
    """
    cmd.set('fetch_path', os.path.expanduser('~/pdb'))
    cmd.set('fetch_type_default', 'mmtf')
    obj = randomstring()
    if os.path.exists(pdb):
        if verbose:
            print(f'Loading from local file {pdb}')
        cmd.load(pdb, object=obj)
    else:
        if verbose:
            print(f'Fetching {pdb} from the PDB')
        cmd.fetch(pdb, name=obj)
    selection = f'{obj} and {selection} and present'
    if view is not None:
        viewmat = load_pymol_view(view)
        cmd.transform_selection(selection, viewmat)
    if not split_by_chains:
        coords = cmd.get_coords(selection=selection)
    else:
        chain_list = cmd.get_chains(selection=selection)
        coords = []
        for chain in chain_list:
            coords.append(cmd.get_coords(selection=f'{selection} and chain {chain}'))
    if not return_selection:
        return coords
    else:
        return coords, selection


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
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
