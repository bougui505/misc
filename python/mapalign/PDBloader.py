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
from misc.mapalign import mapalign
import logging
# from misc.pytorch import torchify


def collate_fn(batch):
    return batch


class PDBdataset(torch.utils.data.Dataset):
    """
    Load pdb files from a PDB database and return coordinates
    See: ~/source/misc/shell/updatePDB.sh to download the PDB

    >>> cmd.reinitialize()
    >>> cmd.load('data/3u97_A.pdb', 'A_')
    >>> coords_a = cmd.get_coords('A_ and polymer.protein and chain A and name CA')
    >>> dmat_a = mapalign.get_dmat(coords_a)
    >>> cmap_a = mapalign.get_cmap(dmat_a)
    >>> # Generate random datapoints for testing
    >>> dataset = PDBdataset('/media/bougui/scratch/pdb', cmap_a, logfilename='mapalign_batch.log')
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)
    >>> for i, batch in enumerate(dataloader):
    ...     print(len(batch))
    ...     # print([coords.shape for coords in batch])
    ...     if i == 0:
    ...         break
    1
    >>> batch
    """
    def __init__(self,
                 pdbpath,
                 cmap_a,
                 logfilename,
                 selection='polymer.protein and name CA',
                 sep_x_list=[2],
                 sep_y_list=[16],
                 gap_e_list=[-0.001]):
        logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
        logging.info(f"################ Starting {__file__} ################")
        logging.info(f"pdbname: score native_contact_ratio")
        self.list_IDs = glob.glob(f'{pdbpath}/**/*.ent.gz')
        self.selection = selection
        self.cmap_a = cmap_a
        self.sep_x_list = sep_x_list
        self.sep_y_list = sep_y_list
        self.gap_e_list = gap_e_list
        cmd.reinitialize()

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        pdbfile = self.list_IDs[index]
        pymolname = randomgen.randomstring()
        cmd.load(filename=pdbfile, object=pymolname)
        chains = cmd.get_chains(pymolname)
        scores = []
        for chain in chains:
            coords = cmd.get_coords(selection=f'{pymolname} and {self.selection} and chain {chain}')
            cmap = mapalign.get_cmap(mapalign.get_dmat(coords))
            aln, score, sep_x, sep_y, gap_e = mapalign.mapalign(self.cmap_a,
                                                                cmap,
                                                                sep_x_list=self.sep_x_list,
                                                                sep_y_list=self.sep_y_list,
                                                                gap_e_list=self.gap_e_list,
                                                                progress=False)
            native_contacts_score = mapalign.get_score(self.cmap_a, cmap, aln)
            scores.append((pdbfile, chain, score, native_contacts_score))
            logging.info(f'{pdbfile}_{chain}: {score:.4f} {native_contacts_score:.4f}')
        cmd.delete(pymolname)
        return scores


if __name__ == '__main__':
    import sys
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
