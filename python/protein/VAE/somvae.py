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

from quicksom import som as quicksom
import PDBloader
import torch
import os
import numpy as np
from misc.eta import ETA
from vae import load_model, forward_batch


def fit(pdbpath=None,
        pdblist=None,
        model=None,
        n_epochs=20,
        batch_size=4,
        num_workers=None,
        dobreak=np.inf,
        print_each=100,
        somsize=(50, 50),
        latent_dims=512,
        outfilename='models/som.p'):
    """
    >>> model = load_model('models/test.pt')
    >>> fit(pdbpath='/media/bougui/scratch/pdb', model=model, n_epochs=1, dobreak=12, print_each=1, outfilename='models/somtest.p')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.eval()
    model = model.to(device)
    som = quicksom.SOM(m=somsize[0], n=somsize[1], dim=latent_dims, n_epoch=n_epochs, device=device)
    if num_workers is None:
        num_workers = os.cpu_count()
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist, interpolate=False, return_name=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    nbatch = len(dataloader)
    if som.alpha is None:
        som.alpha = float((som.m * som.n) / nbatch)
    log(f'som.alpha: {som.alpha}')
    log(f'som.sigma: {som.sigma}')
    dataiter = iter(dataloader)
    epoch = 0
    som.step = 0
    total_steps = n_epochs * len(dataiter)
    eta = ETA(total_steps=total_steps)
    while epoch < n_epochs:
        if som.step >= dobreak:
            break
        try:
            som.step += 1
            lr_step = som.scheduler(som.step, total_steps)
            data = next(dataiter)
            data = [e for e in data if e is not None]
            names = [name for dmat, name in data if dmat is not None]
            batch = [dmat.to(device) for dmat, name in data if dmat is not None]
            if len(batch) > 0:
                with torch.no_grad():
                    _, latent = forward_batch(batch, model, encode_only=True)
                # latent = latent.to(som.device, non_blocking=True)
                # latent = latent.float()
                bmu_loc, error = som.__call__(latent, learning_rate_op=lr_step)
            if not som.step % print_each:
                eta_val = eta(som.step)
                log(f"epoch: {epoch+1}|step: {som.step}|alpha: {som.alpha_op:4f}|sigma: {som.sigma_op:.4f}|error: {error:.4f}|eta: {eta_val}"
                    )
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1
    som.save_pickle(outfilename)


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
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--vae', help='Filename for the trained VAE model')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--pdbpath', help='Path to the PDB database')
    parser.add_argument('--batch_size', help='Batch size for training (default 4)', default=4, type=int)
    parser.add_argument('--print_each', type=int, default=100)
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--somx', help='size of x axis of the SOM', type=int, default=50)
    parser.add_argument('--somy', help='size of y axis of the SOM', type=int, default=50)
    parser.add_argument('--somfile', help='SOM file name to dump')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    model = load_model(args.vae)
    fit(pdbpath=args.pdbpath,
        pdblist=None,
        model=model,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=None,
        dobreak=np.inf,
        print_each=args.print_each,
        somsize=(args.somx, args.somy),
        latent_dims=args.latent_dims,
        outfilename=args.somfile)
