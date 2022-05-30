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

import torch
import time
import datetime
from misc import randomgen
from misc.protein.interpred import utils
from misc.protein.interpred import PDBloader
from misc.protein.interpred import vae
import os
from misc.eta import ETA
import copy
import numpy as np
import DB


def save_model(interpred, filename):
    torch.save(interpred.state_dict(), filename)


def load_model(filename, latent_dims=512):
    """
    # >>> interpred = load_model('models/test.pth')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = vae.VariationalAutoencoder(latent_dims=latent_dims)
    model.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    model.eval()
    return model


def learn(
        dbpath=None,
        pdblist=None,
        nepoch=10,
        batch_size=4,
        num_workers=None,
        print_each=100,
        modelfilename='models/interpred.pt',
        save_each=30,  # in minutes
        latent_dims=512,
        save_each_epoch=True):
    """
    Uncomment the following to test it (about 20s runtime)
    # >>> learn(pdblist=['data/1ycr.pdb'], print_each=1, nepoch=60, modelfilename='models/test.pt', batch_size=1, save_each_epoch=False)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if not os.path.exists(modelfilename):
        model = vae.VariationalAutoencoder(latent_dims=latent_dims).to(device)
    else:
        msg = f'# Loading model: {modelfilename}'
        # print(msg)
        log(msg)
        model = load_model(modelfilename).to(device)
        model.train()  # set model in train mode
    optimizer = torch.optim.Adam(model.parameters())
    if num_workers is None:
        num_workers = os.cpu_count()
    t_0 = time.time()
    save_model(model, modelfilename)
    dataset = PDBloader.PDBdataset(pdbpath=dbpath, pdblist=pdblist)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    epoch = 0
    step = 0
    total_steps = nepoch * len(dataiter)
    eta = ETA(total_steps=total_steps)
    klw = len(dataiter) / batch_size
    while epoch < nepoch:
        try:
            batch = next(dataiter)
            step += 1
            out, targets = vae.forward_batch(batch, model)
            if len(out) > 0 and len(targets) > 0:
                # zero the parameter gradients
                optimizer.zero_grad()
                loss_rec = get_loss(out, targets)
                loss_kl = model.encoder.kl
                loss = loss_rec + klw * loss_kl
                loss.backward()
                optimizer.step()
                if (time.time() - t_0) / 60 >= save_each:
                    t_0 = time.time()
                    save_model(model, modelfilename)
                if not step % print_each:
                    eta_val = eta(step)
                    last_saved = (time.time() - t_0)
                    last_saved = str(datetime.timedelta(seconds=last_saved))
                    log(f"epoch: {epoch+1}|step: {step}|loss: {loss:.4f}|kl: {loss_kl:.4f}|loss_rec: {loss_rec:.4f}|klw: {klw:.4f}|last_saved: {last_saved}|eta: {eta_val}"
                        )
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1
            if save_each_epoch:
                save_model(model, modelfilename)
    save_model(model, modelfilename)


def todevice(*args, device):
    out = []
    for arg in args:
        out.append(arg.to(device))
    return out


def predict(pdb_a,
            pdb_b,
            sel_a='all',
            sel_b='all',
            model=None,
            modelfilename=None,
            input_size=(224, 224),
            doplot=False):
    """
    >>> intercmap = predict(pdb_a='data/1ycr.pdb', pdb_b='data/1ycr.pdb', sel_a='chain A', sel_b='chain B', modelfilename='models/test.pt', doplot=True)
    >>> intercmap.shape
    (85, 13)
    """
    if modelfilename is not None:
        model = load_model(modelfilename)
    model.eval()
    coords_a, seq_a = utils.get_coords(pdb_a, selection=f'polymer.protein and name CA and {sel_a}', return_seq=True)
    coords_b, seq_b = utils.get_coords(pdb_b, selection=f'polymer.protein and name CA and {sel_b}', return_seq=True)
    log(f'coords_a.shape: {coords_a.shape}')
    log(f'coords_b.shape: {coords_b.shape}')
    na = coords_a.shape[1]
    nb = coords_b.shape[1]
    inp, normalizer = utils.get_input(coords_a[0], coords_b[0], input_size=input_size, return_normalizer=True)
    out = model(inp)
    out = torch.nn.functional.interpolate(out, size=(na, nb))
    out, = normalizer.inverse_transform([out])
    log(f'out.shape: {out.shape}')
    intercmap = torch.squeeze(out)
    intercmap = intercmap.detach().cpu().numpy()
    if doplot:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        target = utils.get_inter_dmat(coords_a, coords_b)
        target = torch.squeeze(target.detach().cpu()).numpy()
        m1 = ax1.matshow(target)
        ax1.set_title('Original')
        fig.colorbar(m1, ax=ax1, shrink=.5)
        m2 = ax2.matshow(intercmap)
        ax2.set_title('Reconstructed')
        fig.colorbar(m2, ax=ax2, shrink=.5)
        plt.show()
    return intercmap


def get_loss(out_batch, targets):
    """
    # >>> dataset = PDBloader.PDBdataset(pdblist=['data/1ycr.pdb'], randomize=False)

    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/dimerdb', randomize=False)

    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> len(batch)
    4
    >>> [(inp.shape, intercmap.shape) for (inp, intercmap) in batch]
    [(torch.Size([1, 2, 224, 224]), torch.Size([1, 1, 639, 639])), (torch.Size([1, 2, 224, 224]), torch.Size([1, 1, 339, 339])), (torch.Size([1, 2, 224, 224]), torch.Size([1, 1, 491, 491])), (torch.Size([1, 2, 224, 224]), torch.Size([1, 1, 1323, 57]))]
    >>> model = vae.VariationalAutoencoder(latent_dims=512)
    >>> out, targets = vae.forward_batch(batch, model)
    >>> len(out)
    4
    >>> [o.shape for o in out]
    [torch.Size([1, 1, 639, 639]), torch.Size([1, 1, 339, 339]), torch.Size([1, 1, 491, 491]), torch.Size([1, 1, 1323, 57])]
    >>> [e.shape for e in targets]
    [torch.Size([1, 1, 639, 639]), torch.Size([1, 1, 339, 339]), torch.Size([1, 1, 491, 491]), torch.Size([1, 1, 1323, 57])]
    >>> loss = get_loss(out, targets)
    >>> loss
    tensor(..., grad_fn=<DivBackward0>)
    """
    n = len(targets)
    loss = 0.
    for i in range(n):
        out = out_batch[i][0]
        out = out.float()
        target = targets[i].float()
        loss += ((out - target)**2).mean()
    loss = loss / n
    return loss


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    from pymol import cmd
    import matplotlib.pyplot as plt  # For DOCTESTS
    # ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--train', help='Train the interpred model', action='store_true')
    parser.add_argument('--nepoch', help='Number of epochs for training (default 10)', default=10, type=int)
    parser.add_argument('--batch_size', help='Batch size for training (default 4)', default=4, type=int)
    parser.add_argument('--dbpath', help='Path to the learning database')
    parser.add_argument('--print_each', help='Print each given steps in log file', default=100, type=int)
    parser.add_argument('--save_every',
                        help='Save the model every given number of minutes (default: 30 min)',
                        type=int,
                        default=30)
    parser.add_argument('--predict',
                        help='Predict for the given pdb files (see pdb1, pdb2 options)',
                        action='store_true')
    parser.add_argument('--pdb1', help='First PDB for predicrion')
    parser.add_argument('--pdb2', help='Second PDB for predicrion')
    parser.add_argument('--sel1', help='First selection', default='all')
    parser.add_argument('--sel2', help='Second selection', default='all')
    # parser.add_argument('--ground_truth', help='Compute ground truth from the given pdb', action='store_true')
    parser.add_argument('--model', help='pth filename for the model to load', default=None)
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.train:
        current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        if args.model is None:
            modelfilename = f'models/interpred_{current_time}.pth'
        else:
            modelfilename = args.model
        learn(dbpath=args.dbpath,
              nepoch=args.nepoch,
              batch_size=args.batch_size,
              num_workers=None,
              print_each=args.print_each,
              modelfilename=modelfilename,
              save_each=args.save_every)
    if args.predict:
        intercmap = predict(pdb_a=args.pdb1,
                            pdb_b=args.pdb2,
                            sel_a=args.sel1,
                            sel_b=args.sel2,
                            modelfilename=args.model,
                            doplot=True)
