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
import time
from misc.eta import ETA
import datetime
import utils
import numpy as np

# See: https://discuss.pytorch.org/t/runtimeerror-received-0-items-of-ancdata/4999
torch.multiprocessing.set_sharing_strategy('file_system')


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
    >>> model = encoder.CNN(latent_dims=512)
    >>> out = forward_batch(batch, model)
    >>> [(z_full.shape, z_fragment.shape) for z_full, z_fragment in out]
    [(torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512]))]
    """
    out = []
    for full, fragment in batch:
        try:
            z_full = model(full)
            z_fragment = model(fragment)
            out.append((z_full, z_fragment))
        except RuntimeError:
            print(f'RuntimeError: CUDA out of memory. Will skip this data point of shape: {full.shape}')
            pass
    return out


def get_contrastive_loss(out, tau=1.):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.CNN(latent_dims=512)
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
                # log(f'z_full_i: {z_full_i}')
                # log(f'sim_num: {sim_num}')
                num = torch.exp(sim_num / tau)
            else:
                sim_den = torch.matmul(z_full_i, z_full_j.T)
                # log(f'sim_den: {sim_den}')
                den += torch.exp(sim_den / tau)
        # log(f'num:{num}, den: {den}')
        loss -= torch.log((num + 1e-8) / (den + 1e-8))
    if n > 0:
        loss = loss / n
    loss = torch.squeeze(loss)
    assert not torch.isnan(loss), 'ERROR: loss is nan'
    return loss


def get_norm(out):
    """
    >>> batch = get_batch_test()
    >>> model = encoder.CNN(latent_dims=512)
    >>> out = forward_batch(batch, model)
    >>> [(z_full.shape, z_fragment.shape) for z_full, z_fragment in out]
    [(torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512])), (torch.Size([1, 512]), torch.Size([1, 512]))]
    >>> get_norm(out)
    tensor(..., grad_fn=<MeanBackward0>)
    """
    z_full_list = torch.cat([e[0] for e in out])  # torch.Size([4, 512])
    z_fragment_list = torch.cat([e[1] for e in out])  # torch.Size([4, 512])
    z = torch.cat((z_full_list, z_fragment_list))  # torch.Size([8, 512])
    norms = torch.linalg.norm(z, dim=1)
    return norms.mean()


def save_model(model, filename):
    torch.save(model.state_dict(), filename)


class Metric(object):
    """
    >>> metric = Metric()
    >>> model = encoder.FCN(latent_dims=128)
    >>> metric.get(model)
    ...
    """
    def __init__(self,
                 pdblist1=['pdb/uc/pdb4ucc.ent.gz'],
                 pdblist2=['pdb/wj/pdb2wj8.ent.gz'],
                 sellist1=['chain A'],
                 sellist2=['chain A']):
        self.dmat1list = []
        self.dmat2list = []
        for pdb1, sel1, pdb2, sel2 in zip(pdblist1, sellist1, pdblist2, sellist2):
            coords1 = utils.get_coords(pdb1, sel=sel1)
            coords2 = utils.get_coords(pdb2, sel=sel2)
            self.dmat1list.append(utils.get_dmat(coords1[None, ...]))
            self.dmat2list.append(utils.get_dmat(coords2[None, ...]))

    def get(self, model):
        simlist = []
        for dmat1, dmat2 in zip(self.dmat1list, self.dmat2list):
            with torch.no_grad():
                z1 = model(dmat1)
                z2 = model(dmat2)
            simlist.append(float(torch.matmul(z1, z2.T).squeeze().numpy()))
        sim = np.mean(simlist)
        return sim


def train(
        pdbpath=None,
        pdblist=None,
        batch_size=4,
        n_epochs=20,
        latent_dims=128,
        save_each_epoch=True,
        print_each=100,
        save_each=30,  # in minutes
        input_size=(224, 224),
        modelfilename='models/sscl.pt',
        cnn=True):
    """
    # >>> train(pdbpath='pdb', print_each=1, save_each_epoch=False, n_epochs=3, modelfilename='models/1.pt', batch_size=32)
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
        if cnn:
            log('CNN model')
            model = encoder.CNN(latent_dims=latent_dims, input_size=input_size)
        else:
            log('FCN model')
            model = encoder.FCN(latent_dims=latent_dims)
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters())
    t_0 = time.time()
    save_model(model, modelfilename)
    epoch = 0
    step = 0
    metric = Metric()
    total_steps = n_epochs * len(dataiter)
    eta = ETA(total_steps=total_steps)
    while epoch < n_epochs:
        try:
            step += 1
            batch = next(dataiter)
            batch = [(e[0].to(device), e[1].to(device)) for e in batch if e is not None]
            bs = len(batch)
            if bs > 0:
                opt.zero_grad()
                out = forward_batch(batch, model)
                if len(out) > 0:
                    loss = get_contrastive_loss(out)
                    loss.backward()
                    # torch.nn.utils.clip_grad_value_(model.parameters(), clip_value=1.0)
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
                    opt.step()
            if (time.time() - t_0) / 60 >= save_each:
                t_0 = time.time()
                save_model(model, modelfilename)
            if not step % print_each:
                eta_val = eta(step)
                last_saved = (time.time() - t_0)
                last_saved = str(datetime.timedelta(seconds=last_saved))
                norm = get_norm(out)
                metricval = metric.get(model)
                if loss is not None:
                    log(f"epoch: {epoch+1}|step: {step}|loss: {loss:.4f}|metric: {metricval:.4f}|norm: {norm:.4f}|bs: {bs}|last_saved: {last_saved}| eta: {eta_val}"
                        )
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
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--train', help='Train the SSCL', action='store_true')
    parser.add_argument('--bs', help='Batch size', type=int, default=4)
    parser.add_argument('--pdbpath', help='Path to the PDB database')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--print_each', type=int, default=100)
    parser.add_argument('--save_every',
                        help='Save the model every given number of minutes (default: 30 min)',
                        type=int,
                        default=30)
    parser.add_argument('--model', help='Model to load or for saving', metavar='model.pt')
    parser.add_argument('--latent_dims', default=128, type=int)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--model_type', help='model type to use. Can be CNN (default) or FCN', default='CNN')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    input_size = (args.input_size, args.input_size)
    if args.train:
        if args.model_type == 'CNN':
            cnn = True
        else:
            cnn = False
        train(pdbpath=args.pdbpath,
              n_epochs=args.epochs,
              modelfilename=args.model,
              print_each=args.print_each,
              latent_dims=args.latent_dims,
              save_each=args.save_every,
              batch_size=args.bs,
              input_size=input_size,
              cnn=cnn)
