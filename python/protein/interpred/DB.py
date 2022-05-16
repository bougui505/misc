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

import PDBloader
import torch
from misc.eta import ETA
import numpy as np
import os
import glob


def prepare_db(outpath, pdbpath=None, pdblist=None, num_workers=None, dobreak=np.inf, print_each=100):
    """
    >>> prepare_db(outpath='/media/bougui/scratch/pdb_cmaps', pdbpath='/media/bougui/scratch/pdb', print_each=1, dobreak=100)
    """
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist, randomize=False, return_name=True)
    if num_workers is None:
        num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    total_steps = np.min([len(dataiter), dobreak])
    log(f'total_steps: {total_steps}')
    eta = ETA(total_steps=total_steps)
    step = 0
    try:
        os.mkdir(outpath)
    except FileExistsError:
        pass
    while True:
        try:
            batch = next(dataiter)
            cmap_a, cmap_b, interseq, intercmap, name = process_batch(batch)
            directory = os.path.basename(os.path.dirname(name))
            outdir = f'{outpath}/{directory}'
            name = os.path.splitext(os.path.splitext(os.path.basename(name))[0])[0]
            try:
                os.mkdir(outdir)
            except FileExistsError:
                pass
            if cmap_a is not None:
                cmap_a = torch.nn.functional.interpolate(cmap_a, size=224)
                cmap_b = torch.nn.functional.interpolate(cmap_b, size=224)
                interseq = torch.nn.functional.interpolate(interseq, size=224)
                np.savez_compressed(f'{outdir}/{name}.cmap.npz',
                                    cmap_a=cmap_a,
                                    cmap_b=cmap_b,
                                    interseq=interseq,
                                    intercmap=intercmap)
            step += 1
            if step >= dobreak:
                break
            if not step % print_each:
                eta_val = eta(step)
                log(f"step: {step}|directory: {directory}|name: {name}|eta: {eta_val}")
        except StopIteration:
            break
        except AssertionError:
            pass


def collate_fn(batch):
    cmap_a = torch.cat([torch.tensor(e[0]) for e in batch])
    cmap_b = torch.cat([torch.tensor(e[1]) for e in batch])
    interseq = torch.cat([torch.tensor(e[2]) for e in batch])
    intercmap = [torch.tensor(e[3]) for e in batch]
    return cmap_a, cmap_b, interseq, intercmap


class CmapDataset(torch.utils.data.Dataset):
    """
    >>> cmapds = CmapDataset('/media/bougui/scratch/pdb_cmaps')
    >>> dataloader = torch.utils.data.DataLoader(cmapds, batch_size=4, shuffle=True, num_workers=4, collate_fn=collate_fn)
    >>> batch = next(iter(dataloader))
    >>> cmap_a, cmap_b, interseq, intercmap = batch
    >>> cmap_a.shape, cmap_b.shape, interseq.shape, len(intercmap)
    (torch.Size([4, 1, 224, 224]), torch.Size([4, 1, 224, 224]), torch.Size([4, 42, 224, 224]), 4)
    """
    def __init__(self, cmapdbpath):
        self.list_IDs = glob.glob(f'{cmapdbpath}/**/*.cmap.npz')

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        npzfilename = self.list_IDs[index]
        data = np.load(npzfilename)
        cmap_a = data['cmap_a']
        cmap_b = data['cmap_b']
        interseq = data['interseq']
        intercmap = data['intercmap']
        return cmap_a, cmap_b, interseq, intercmap


def process_batch(batch):
    coords_a, coords_b, interseq, cmap, name = batch[0]
    if coords_a is not None:
        cmap_a, cmap_b = get_input_mats(coords_a[0], coords_b[0])
    else:
        return None, None, None, None, name
    return cmap_a, cmap_b, interseq, cmap[0, 0, ...], name


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    from interpred import get_input_mats
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='Path to the pdb database to process')
    parser.add_argument('--out', help='Path to the directory to store the DB')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    prepare_db(args.out, pdbpath=args.pdb, pdblist=None, num_workers=None, dobreak=np.inf, print_each=100)
