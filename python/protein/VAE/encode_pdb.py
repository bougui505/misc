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
import os
import numpy as np
import vae
import cmapvae
import faiss
from misc.eta import ETA


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def encode_pdb(pdbfilelist, model, indexfilename, batch_size=4, do_break=np.inf, latent_dims=512):
    """
    >>> model = cmapvae.load_model('models/cmapvae_20220525_0843.pt')
    >>> encode_pdb('pdbfilelist.txt', model, indexfilename='index.faiss', do_break=3)
    Total number of pdb in the FAISS index: 12
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        os.mkdir(indexfilename)
    except FileExistsError:
        pass
    model = model.to(device)
    model.eval()
    index = faiss.IndexFlatL2(latent_dims)  # build the index
    dataset = PDBloader.PDBdataset(pdblistfile=pdbfilelist, return_name=True, interpolate=False)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    names = []
    eta = ETA(total_steps=len(dataloader))
    for i, data in enumerate(dataloader):
        if i >= do_break:
            break
        batch = [dmat.to(device) for dmat, name in data if dmat is not None]
        names.extend([name for dmat, name in data if dmat is not None])
        with torch.no_grad():
            _, latent_vectors = vae.forward_batch(batch, model, encode_only=True)
        index.add(latent_vectors.detach().cpu().numpy())
        eta_val = eta(i + 1)
        log(f"step: {i+1}|eta: {eta_val}")
    npdb = index.ntotal
    print(f'Total number of pdb in the FAISS index: {npdb}')
    faiss.write_index(index, f'{indexfilename}/{indexfilename}')
    names = np.asarray(names)
    np.save(f'{indexfilename}/ids.npy', names)


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        '--pdbfilelist',
        help=
        'Line formatted test file with the name of the pdbfile and the chain with this line format: "/path/to/file.pdb chain"'
    )
    parser.add_argument('--model', help='Model to load or for saving', metavar='model.pt')
    parser.add_argument('--index', help='name of the output faiss index filename')
    parser.add_argument('--batch_size', help='Batch size (default 4)', default=4, type=int)
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    model = cmapvae.load_model(filename=args.model, latent_dims=args.latent_dims)
    encode_pdb(args.pdbfilelist, model, args.index, batch_size=args.batch_size, latent_dims=args.latent_dims)
