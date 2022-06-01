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


class Properties(object):
    def __init__(self, pymolname):
        self.mol = pymolname

    def n_chains(self):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        n_chains = len(chains)
        return n_chains

    def n_contacts(self):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        nchains = len(chains)
        total = 0
        if nchains > 1:
            for i in range(nchains):
                for j in range(i + 1, nchains):
                    c1, c2 = chains[i], chains[j]
                    _, ncontacts = self.isinteracting(c1, c2)
                    total += ncontacts
        return total

    def nres(self):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        nres = []
        for chain in chains:
            coords = cmd.get_coords(f'{self.mol} and polymer.protein and name CA and chain {chain}')
            if coords is not None:
                nres.append(len(coords))
            else:
                nres.append(0)
        return nres

    def chains(self):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        return chains

    def isinteracting(self, chain1, chain2, distance_thr=8.):
        coords1 = cmd.get_coords(f'{self.mol} and polymer.protein and name CA and chain {chain1}')
        coords2 = cmd.get_coords(f'{self.mol} and polymer.protein and name CA and chain {chain2}')
        if coords1 is None or coords2 is None:
            return False, 0
        if coords1.ndim == 2 and coords2.ndim == 2:
            interdist = scidist.cdist(coords1, coords2)
            contacts = interdist < distance_thr
            ncontacts = contacts.sum()
            return contacts.any(), ncontacts
        else:
            return False, 0

    def checkseq(self):
        chains = cmd.get_chains(f'{self.mol} and polymer.protein')
        seq = ''
        coords = cmd.get_coords(f'{self.mol} and polymer.protein and name CA')
        if coords is None:
            return False
        for chain in chains:
            seq += get_seq(self.mol, selection=f'polymer.protein and name CA and chain {chain}')
        return len(seq) == len(coords)


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    """
    def __init__(self,
                 pdbpath,
                 selection='all',
                 return_name=False,
                 feature_list=['seqOK', 'n_chains', 'chains', 'ncontacts', 'nres']):
        self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        self.return_name = return_name
        self.selection = selection
        cmd.reinitialize()
        self.logfilename = logging.getLogger().handlers[0].baseFilename
        self.logfile = open(self.logfilename, 'r')
        self.feature_list = feature_list

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        self.logfile.seek(0)
        features = {}
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        properties = Properties(pymolname)
        if 'seqOK' in self.feature_list:
            features['seqOK'] = properties.checkseq()
        if 'n_chains' in self.feature_list:
            features['n_chains'] = properties.n_chains()
        if 'chains' in self.feature_list:
            chains = properties.chains()
            features['chains'] = ','.join([f'{e}' for e in chains])
        if 'ncontacts' in self.feature_list:
            features['ncontacts'] = properties.n_contacts()
        if 'nres' in self.feature_list:
            nres = properties.nres()
            features['nres'] = ','.join([f'{e:d}' for e in nres])
        outstr = "|".join([f'{k}: {v}' for k, v in features.items()])
        log(f'pdbfile: {pdbfile}|{outstr}')
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
    feature_list = ['seqOK', 'n_chains', 'chains', 'ncontacts', 'nres']
    parser.add_argument('--features',
                        help=f'list of features to return. Can be: {feature_list}',
                        nargs='+',
                        default=feature_list)
    args = parser.parse_args()
    # ### UNCOMMENT FOR LOGGING ####
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####

    if args.test:
        doctest.testmod()
        sys.exit()
    dataset = PDBdataset(args.pdb, feature_list=args.features)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=os.cpu_count(),
                                             collate_fn=collate_fn)
    pbar = tqdm.tqdm(total=len(dataset))
    for pdb in dataloader:
        pbar.update(1)
