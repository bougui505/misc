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

from misc.protein.sscl import PDBloader
from misc.protein.sscl import encoder
from misc.protein.sscl.encoder import load_model
import torch
import os


def get_batch_test():
    """
    >>> batch = get_batch_test()
    >>> len(batch)
    4
    >>> [(dmat.shape, dmat_fragment.shape) for dmat, dmat_fragment in batch]
    [(torch.Size([1, 1, 249, 249]), torch.Size([1, 1, ..., ...])), (torch.Size([1, 1, 639, 639]), torch.Size([1, 1, ..., ...])), (torch.Size([1, 1, 390, 390]), torch.Size([1, 1, ..., ...])), (torch.Size([1, 1, 131, 131]), torch.Size([1, 1, ..., ...]))]
    """
    dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb')
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=4,
                                             shuffle=False,
                                             num_workers=2,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    batch = next(dataiter)
    return batch


def forward_batch(batch, model):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.Encoder(latent_dims=512)
    >>> out = forward_batch(batch, model)
    >>> [(z_full.shape, z_fragment.shape) for z_full, z_fragment in out]
    [(torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512]))]
    """
    out = []
    for full, fragment in batch:
        z_full = model(full)
        z_fragment = model(fragment)
        out.append((z_full, z_fragment))
    return out


def get_contrastive_loss(out, tau=1.):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.Encoder(latent_dims=512)
    >>> out = forward_batch(batch, model)
    >>> [(z_full.shape, z_fragment.shape) for z_full, z_fragment in out]
    [(torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512]))]
    >>> loss = get_contrastive_loss(out)
    >>> loss
    tensor(..., grad_fn=<SqueezeBackward0>)
    """
    n = len(out)
    z_full_list = [e[0] for e in out]
    z_fragment_list = [e[1] for e in out]
    loss = 0.
    for i in range(n):
        z_full_i = z_full_list[i]
        # z_fragment_i = z_fragment_list[i]
        den = 0.
        for j in range(n):
            z_full_j = z_full_list[j]
            z_fragment_j = z_fragment_list[j]
            if i == j:
                sim_num = torch.matmul(z_full_i, z_fragment_j.T)
                num = torch.exp(sim_num / tau)
            else:
                sim_den = torch.matmul(z_full_i, z_full_j.T)
                den += torch.exp(sim_den / tau)
        loss -= torch.log(num / den)
    loss = torch.squeeze(loss)
    return loss


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


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
        input_size=(224, 224),
        klwscheduler=False):
    """
    >>> train(pdblist=['data/1ycr.pdb'], print_each=1, save_each_epoch=False, n_epochs=3, modelfilename='models/1.pt')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist)
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
        model = encoder.Encoder(latent_dims=latent_dims, input_size=input_size)
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
            normalizer = Normalizer(batch)
            batch = normalizer.transform(normalizer.batch)
            bs = len(batch)
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1
            if save_each_epoch:
                t_0 = time.time()
                save_model(model, modelfilename)


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
