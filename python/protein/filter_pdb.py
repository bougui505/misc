#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
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

import torch
import glob
from pymol import cmd
from misc import randomgen
import os
import scipy.spatial.distance as scidist


def collate_fn(batch):
    return batch


def get_seq(pymolname, selection):
    seq = cmd.get_fastastr(f'{pymolname} and {selection} and present')
    seq = seq.split()[1:]
    seq = ''.join(seq)
    seq = seq.upper()
    return seq


class Filter(object):
    def __init__(self, pymolname):
        self.mol = pymolname

    def n_chains(self, n=None, lower=None, upper=None):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        n_chains = len(chains)
        if n is not None:
            return n_chains == n

    def isinteracting(self, chain1, chain2, distance_thr=8.):
        coords1 = cmd.get_coords(f'{self.mol} and polymer.protein and name CA and chain {chain1}')
        coords2 = cmd.get_coords(f'{self.mol} and polymer.protein and name CA and chain {chain2}')
        interdist = scidist.cdist(coords1, coords2)
        contacts = interdist < distance_thr
        ncontacts = contacts.sum()
        return contacts.any(), ncontacts

    def checkseq(self):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        seq = ''
        coords = cmd.get_coords(f'{self.mol} and polymer.protein and name CA')
        if coords is None:
            return False
        for chain in chains:
            seq += get_seq(self.mol, selection=f'polymer.protein and name CA and chain {chain}')
        return len(seq) == len(coords)

    def isdimer(self):
        """
        2 chains in contact
        """
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        n_chains = len(chains)
        if n_chains == 2:
            isinter, ncontacts = self.isinteracting(chains[0], chains[1])
            if ncontacts >= 10:
                return isinter, n_chains, ncontacts
            else:
                return False, n_chains, ncontacts
        else:
            return False, n_chains, None


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    """
    def __init__(self, pdbpath, selection='all', return_name=False, nchains=None, dimer=False):
        self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        log('#file #nchains #n_contacts')
        self.return_name = return_name
        self.nchains = nchains
        self.dimer = dimer
        self.selection = selection
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        pdbfilter = Filter(pymolname)
        outfile = None
        if self.nchains is not None:
            if pdbfilter.n_chains(self.nchains):
                outfile = pdbfile
        if self.dimer is not None:
            test, nchains, ncontacts = pdbfilter.isdimer()
            if test:
                if pdbfilter.checkseq():
                    outfile = pdbfile
        if outfile is not None:
            log(f'{outfile} {nchains} {ncontacts}')
        cmd.delete(pymolname)


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import tqdm
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--pdb', help='Path to the pdb database')
    parser.add_argument('--nchains', help='Retrieve pdb with this exact number of chains', type=int)
    parser.add_argument('--dimer', help='Returns only dimers', action='store_true')
    args = parser.parse_args()
    # ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####

    if args.test:
        doctest.testmod()
        sys.exit()
    dataset = PDBdataset(args.pdb, nchains=args.nchains)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=os.cpu_count(),
                                             collate_fn=collate_fn)
    pbar = tqdm.tqdm(total=len(dataset))
    for pdb in dataloader:
        pbar.update(1)
