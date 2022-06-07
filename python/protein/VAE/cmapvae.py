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

import misc.protein.VAE.vae as vae
import misc.protein.VAE.PDBloader as PDBloader
import torch
import os
from misc.eta import ETA
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from misc.protein.VAE.utils import Normalizer
from misc.protein.VAE.vae import forward_batch, load_model


def train(
        pdbpath=None,
        pdblist=None,
        batch_size=4,
        n_epochs=20,
        latent_dims=512,
        save_each_epoch=True,
        print_each=100,
        save_each=30,  # in minutes
        modelfilename='models/cmapvae.pt',
        klwscheduler=False,
        input_size=(512, 512),
        normalize=True):
    """
    >>> train(pdblist=['data/1ycr.pdb'], print_each=1, save_each_epoch=False, n_epochs=3, modelfilename='models/1.pt')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist, interpolate=False)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    if os.path.exists(modelfilename):
        log(f'# Loading model from {modelfilename}')
        model = load_model(filename=modelfilename, latent_dims=latent_dims)
        model.train()
    else:
        model = vae.VariationalAutoencoder(latent_dims=latent_dims, input_size=input_size)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    t_0 = time.time()
    save_model(model, modelfilename)
    epoch = 0
    step = 0
    total_steps = n_epochs * len(dataiter)
    eta = ETA(total_steps=total_steps)
    while epoch < n_epochs:
        try:
            step += 1
            batch = next(dataiter)
            batch = [e.to(device) for e in batch if e is not None]
            if normalize:
                normalizer = Normalizer(batch)
                batch = normalizer.transform(normalizer.batch)
            bs = len(batch)
            if klwscheduler:
                klw = torch.sigmoid(torch.tensor(step * 10 / total_steps - 5.))
            else:
                # See: https://qr.ae/pGSwAn and https://arxiv.org/pdf/1312.6114.pdf
                klw = len(dataiter) / batch_size
            if bs > 0:
                opt.zero_grad()
                model.encoder.reset_kl()
                inputs, outputs = forward_batch(batch, model)
                loss_rec = get_reconstruction_loss(inputs, outputs)
                loss_kl = torch.mean(torch.tensor(model.encoder.kl))
                loss = loss_rec + klw * loss_kl
                if torch.isnan(loss):
                    log('NAN-loss break!')
                    break
                loss.backward()
                opt.step()
            if (time.time() - t_0) / 60 >= save_each:
                t_0 = time.time()
                save_model(model, modelfilename)
            if not step % print_each:
                eta_val = eta(step)
                last_saved = (time.time() - t_0)
                last_saved = str(datetime.timedelta(seconds=last_saved))
                log(f"epoch: {epoch+1}|step: {step}|loss: {loss:.4f}|kl: {loss_kl:.4f}|rec: {loss_rec:.4f}|klw: {klw:.4f}|bs: {bs}|last_saved: {last_saved}| eta: {eta_val}"
                    )
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1
            if save_each_epoch:
                t_0 = time.time()
                save_model(model, modelfilename)
    t_0 = time.time()
    save_model(model, modelfilename)


def plot_reconstructed(pdbfilename, model, selection='polymer.protein and name CA'):
    """
    >>> model = load_model('models/test.pt')
    >>> plot_reconstructed('data/1ycr.pdb', model, selection='polymer.protein and name CA and chain A')
    """
    dataset = PDBloader.PDBdataset(pdblist=[pdbfilename], selection=selection, interpolate=False)
    inp = dataset.__getitem__(0)
    inp = [e[None, ...] for e in inp]
    inp, out = vae.reconstruct(inp, model)
    inp = np.squeeze(inp.detach().cpu().numpy())
    out = np.squeeze(out.detach().cpu().numpy())
    fig, (ax1, ax2) = plt.subplots(1, 2)
    m1 = ax1.matshow(inp)
    ax1.set_title('Original')
    fig.colorbar(m1, ax=ax1, shrink=.5)
    m2 = ax2.matshow(out)
    ax2.set_title('Reconstructed')
    fig.colorbar(m2, ax=ax2, shrink=.5)
    plt.show()


def get_reconstruction_loss(inputs, targets):
    """
    >>> model = vae.VariationalAutoencoder(latent_dims=512)
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb', interpolate=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = dataiter.__next__()
    >>> inputs, outputs = forward_batch(batch, model)
    >>> loss = get_reconstruction_loss(inputs, outputs)
    >>> loss
    tensor(..., grad_fn=<DivBackward0>)
    """
    loss = 0
    n = len(inputs)
    for (i, o) in zip(inputs, targets):
        # loss += ((i - o)**2).mean()
        loss += torch.nn.functional.binary_cross_entropy(i, o)
    loss = loss / n
    return loss


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


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
    parser.add_argument('--train', help='Train the VAE', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--klw', help='Switch on weight scheduler for kl divergence', action='store_true')
    parser.add_argument('--pdbpath', help='Path to the PDB database')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--print_each', type=int, default=100)
    parser.add_argument('--save_every',
                        help='Save the model every given number of minutes (default: 30 min)',
                        type=int,
                        default=30)
    parser.add_argument('--model', help='Model to load or for saving', metavar='model.pt')
    parser.add_argument('--latent_dims', default=512, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--predict', help='Reconstruction from the given pdb', metavar='filename.pdb')
    parser.add_argument('--sel', default='polymer.protein and name CA')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    input_size = (args.input_size, args.input_size)

    if args.train:
        train(pdbpath=args.pdbpath,
              n_epochs=args.epochs,
              modelfilename=args.model,
              print_each=args.print_each,
              latent_dims=args.latent_dims,
              klwscheduler=args.klw,
              save_each=args.save_every,
              batch_size=args.bs,
              input_size=input_size)

    if args.predict is not None:
        model = load_model(args.model, latent_dims=args.latent_dims)
        model.encoder.input_size = input_size
        plot_reconstructed(args.predict, model, selection=args.sel)
