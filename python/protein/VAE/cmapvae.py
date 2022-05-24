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

import vae
import PDBloader
import torch
import os
from misc.eta import ETA
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np


def train(
        pdbpath=None,
        pdblist=None,
        batch_size=4,
        n_epochs=20,
        latent_dims=10,
        save_each_epoch=True,
        print_each=100,
        save_each=30,  # in minutes
        modelfilename='models/cmapvae.pt'):
    """
    >>> train(pdblist=['data/1ycr.pdb'], print_each=1, save_each_epoch=False, n_epochs=200, modelfilename='models/test.pt')
    """
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist)
    num_workers = os.cpu_count()
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    model = vae.VariationalAutoencoder(latent_dims=latent_dims)
    opt = torch.optim.Adam(model.parameters())
    t_0 = time.time()
    save_model(model, modelfilename)
    epoch = 0
    step = 0
    total_steps = n_epochs * len(dataiter)
    eta = ETA(total_steps=total_steps)
    while epoch < n_epochs:
        try:
            batch = next(dataiter)
            step += 1
            opt.zero_grad()
            model.encoder.reset_kl()
            inputs, outputs = forward_batch(batch, model)
            loss_rec = get_reconstruction_loss(inputs, outputs)
            loss_kl = torch.mean(torch.Tensor(model.encoder.kl))
            loss = loss_rec + loss_kl
            loss.backward()
            opt.step()
            if (time.time() - t_0) / 60 >= save_each:
                t_0 = time.time()
                save_model(model, modelfilename)
            if not step % print_each:
                eta_val = eta(step)
                last_saved = (time.time() - t_0)
                last_saved = str(datetime.timedelta(seconds=last_saved))
                log(f"epoch: {epoch+1}|step: {step}|loss: {loss:.4f}|kl: {loss_kl:.4f}|rec: {loss_rec:.4f}|last_saved: {last_saved}| eta: {eta_val}"
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
    dataset = PDBloader.PDBdataset(pdblist=[pdbfilename], selection=selection)
    inp = dataset.__getitem__(0)
    model.eval()
    out = model(inp)
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


def forward_batch(batch, model):
    """
    >>> model = vae.VariationalAutoencoder(latent_dims=10)
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb')
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=2, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = dataiter.__next__()
    >>> inputs, outputs = forward_batch(batch, model)
    >>> [e.shape for e in inputs]
    [torch.Size([1, 1, 249, 249]), torch.Size([1, 1, 639, 639]), torch.Size([1, 1, 390, 390]), torch.Size([1, 1, 131, 131])]
    >>> [e.shape for e in outputs]
    [torch.Size([1, 1, 249, 249]), torch.Size([1, 1, 639, 639]), torch.Size([1, 1, 390, 390]), torch.Size([1, 1, 131, 131])]
    """
    inputs = [e for e in batch if e is not None]
    outputs = []
    for data in inputs:
        out = model(data)
        outputs.append(out)
    return inputs, outputs


def get_reconstruction_loss(inputs, targets):
    """
    >>> model = vae.VariationalAutoencoder(latent_dims=10)
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb')
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
        loss += ((i - o)**2).mean()
    loss = loss / n
    return loss


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


def load_model(filename, latent_dims=10):
    """
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = vae.VariationalAutoencoder(latent_dims=latent_dims)
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    model.eval()
    return model


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
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
