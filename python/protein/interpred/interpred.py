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
from misc import randomgen
from misc.protein.interpred import utils
from misc.protein.interpred import PDBloader
import os
from misc.eta import ETA
import copy
import numpy as np
import SquashLayer


class InterPred(torch.nn.Module):
    """
    >>> coords_A, seq_a = utils.get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA', return_seq=True)
    >>> coords_A.shape
    torch.Size([1, 85, 3])
    >>> coords_B, seq_b = utils.get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA', return_seq=True)
    >>> coords_B.shape
    torch.Size([1, 13, 3])
    >>> interseq = utils.get_inter_seq(seq_a, seq_b)
    >>> interseq.shape
    torch.Size([1, 42, 85, 13])

    >>> interpred = InterPred(verbose=True)
    [Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU()]
    [Conv2d(42, 32, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=same), ReLU(), Conv2d(256, 1, kernel_size=(3, 3), stride=(1, 1), padding=same), Sigmoid()]
    >>> dmat_a, dmat_b, maxab = get_input_mats(coords_A, coords_B)
    >>> interpred(dmat_a, dmat_b, interseq).shape
    torch.Size([1, 85, 13])
    >>> len(list(interpred.parameters()))
    54
    """
    def __init__(self, out_channels=[32, 32, 64, 64, 128, 128, 256, 256], verbose=False):
        super(InterPred, self).__init__()
        in_channels = [1] + out_channels[:-1]
        layers = []
        for i, (ic, oc) in enumerate(zip(in_channels, out_channels)):
            layers.append(torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, padding='same'))
            layers.append(torch.nn.ReLU())
        if verbose:
            print(layers)
        self.fcn_a = torch.nn.Sequential(*layers)
        self.fcn_b = copy.deepcopy(self.fcn_a)
        in_channels = [42] + out_channels[:-1]
        layers_seq = []
        for i, (ic, oc) in enumerate(zip(in_channels, out_channels)):
            layers_seq.append(torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=3, padding='same'))
            layers_seq.append(torch.nn.ReLU())
        layers_seq = layers_seq + [
            torch.nn.Conv2d(in_channels=out_channels[-1], out_channels=1, kernel_size=3, padding='same')
        ]
        layers_seq.append(torch.nn.Sigmoid())
        if verbose:
            print(layers_seq)
        self.fcn_seq = torch.nn.Sequential(*layers_seq)
        self.squashlayer_a = SquashLayer.SquashLayer()
        self.squashlayer_b = SquashLayer.SquashLayer()

    def forward(self, mat_a, mat_b, interseq):
        # batchsize, na, spacedim = coords_a.shape
        out_a = self.fcn_a(mat_a)
        # print(out_a.shape)  # torch.Size([1, 256, 85, 85])
        out_a = self.squashlayer_a(out_a)
        # out_a = out_a.mean(axis=-1)
        out_b = self.fcn_b(mat_b)
        # print(out_b.shape)  # torch.Size([1, 256, 13, 13])
        out_b = self.squashlayer_b(out_b)
        # out_b = out_b.mean(axis=-1)
        out = torch.einsum('ijk,lmn->ikn', out_a, out_b)
        out_seq = self.fcn_seq(interseq)[:, 0, :, :]
        out = out * out_seq
        return out


def get_input_mats(coords_a, coords_b):
    dmat_a = utils.get_dmat(coords_a)
    dmat_b = utils.get_dmat(coords_b)
    maxab = max(dmat_a.max(), dmat_b.max())
    dmat_a = dmat_a / maxab
    dmat_b = dmat_b / maxab
    return dmat_a, dmat_b, maxab


def save_model(interpred, filename):
    torch.save(interpred.state_dict(), filename)


def load_model(filename):
    """
    >>> interpred = load_model('models/test.pth')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interpred = InterPred()
    interpred.load_state_dict(torch.load(filename, map_location=torch.device(device)))
    interpred.eval()
    return interpred


def learn(pdbpath=None,
          pdblist=None,
          nepoch=10,
          batch_size=4,
          num_workers=None,
          print_each=100,
          modelfilename='models/interpred.pth'):
    """
    Uncomment the following to test it (about 20s runtime)
    >>> learn(pdblist=['data/1ycr.pdb'], print_each=1, nepoch=500, modelfilename='models/test.pth')
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    interpred = InterPred().to(device)
    optimizer = torch.optim.Adam(interpred.parameters(), lr=1e-3, weight_decay=1e-5)
    if num_workers is None:
        num_workers = os.cpu_count()
    dataset = PDBloader.PDBdataset(pdbpath=pdbpath, pdblist=pdblist, randomize=True)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             collate_fn=PDBloader.collate_fn)
    dataiter = iter(dataloader)
    epoch = 0
    step = 0
    eta = ETA(total_steps=nepoch * len(dataiter))
    while epoch < nepoch:
        try:
            batch = next(dataiter)
            step += 1
            out, targets = forward_batch(batch, interpred, device=device)
            if len(out) > 0:
                # zero the parameter gradients
                optimizer.zero_grad()
                loss = get_loss(out, targets)
                loss.backward()
                optimizer.step()
                if not step % print_each:
                    eta_val = eta(step)
                    log(f"epoch: {epoch+1}|step: {step}|loss: {loss:.4f}|resolution: {torch.sqrt(loss):.4f} Å|eta: {eta_val}"
                        )
        except StopIteration:
            dataiter = iter(dataloader)
            epoch += 1
            save_model(interpred, modelfilename)
        except:
            print(f'WARNING: Problem for loading at epoch {epoch} and step {step}')
            continue


def predict(pdb_a, pdb_b, sel_a='all', sel_b='all', interpred=None, modelfilename=None):
    """
    >>> intercmap = predict(pdb_a='data/1ycr.pdb', pdb_b='data/1ycr.pdb', sel_a='chain A', sel_b='chain B', modelfilename='models/test.pth')
    >>> intercmap.shape
    (85, 13)
    >>> coords_a = utils.get_coords('data/1ycr.pdb', selection='polymer.protein and chain A and name CA')
    >>> coords_b = utils.get_coords('data/1ycr.pdb', selection='polymer.protein and chain B and name CA')
    >>> target = utils.get_inter_dmat(coords_a, coords_b)
    >>> target = torch.squeeze(target.detach().cpu()).numpy()
    >>> target.shape
    (85, 13)
    >>> get_loss([torch.tensor(intercmap)[None, ...]], [torch.tensor(target)[None, ...]])
    tensor(0.1505)

    >>> fig, axs = plt.subplots(1, 2)
    >>> _ = axs[0].matshow(intercmap)
    >>> _ = axs[0].set_title('Prediction')
    >>> _ = axs[1].matshow(target)
    >>> _ = axs[1].set_title('Ground truth')
    >>> # plt.colorbar()
    >>> plt.show()
    """
    if modelfilename is not None:
        interpred = load_model(modelfilename)
    interpred.eval()
    coords_a, seq_a = utils.get_coords(pdb_a, selection=f'polymer.protein and name CA and {sel_a}', return_seq=True)
    coords_b, seq_b = utils.get_coords(pdb_b, selection=f'polymer.protein and name CA and {sel_b}', return_seq=True)
    interseq = utils.get_inter_seq(seq_a, seq_b)
    dmat_a, dmat_b, maxab = get_input_mats(coords_a, coords_b)
    intercmap = torch.squeeze(interpred(dmat_a, dmat_b, interseq)) * maxab
    # mask = get_mask(intercmap)
    intercmap = intercmap.detach().cpu().numpy()
    # intercmap = np.ma.masked_array(intercmap, mask)
    return intercmap


def forward_batch(batch, interpred, device='cpu'):
    """
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    [(None, None, None, None), (torch.Size([1, 1, 639, 3]), torch.Size([1, 1, 639, 3]), torch.Size([1, 42, 639, 639]), torch.Size([1, 1, 1, 639, 639])), (torch.Size([1, 1, 390, 3]), torch.Size([1, 1, 390, 3]), torch.Size([1, 42, 390, 390]), torch.Size([1, 1, 1, 390, 390])), (None, None, None, None)]
    >>> interpred = InterPred()
    >>> out, targets = forward_batch(batch, interpred)
    >>> len(out)
    2
    >>> [e.shape for e in out]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    >>> [e.shape for e in targets]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    """
    out = []
    targets = []
    for data in batch:
        coords_a, coords_b, interseq, cmap = data
        if coords_a is not None:
            coords_a = coords_a.to(device)
            coords_b = coords_b.to(device)
            interseq = interseq.to(device)
            cmap = cmap.to(device)
            coords_a = coords_a[0]  # Remove the extra dimension not required
            coords_b = coords_b[0]
            dmat_a, dmat_b, maxab = get_input_mats(coords_a, coords_b)
            intercmap = interpred(dmat_a, dmat_b, interseq) * maxab
            out.append(intercmap)
            targets.append(cmap[0, 0, ...])
    return out, targets


def get_loss(out_batch, targets):
    """
    >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=PDBloader.collate_fn)
    >>> dataiter = iter(dataloader)
    >>> batch = next(dataiter)
    >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    [(None, None, None, None), (torch.Size([1, 1, 639, 3]), torch.Size([1, 1, 639, 3]), torch.Size([1, 42, 639, 639]), torch.Size([1, 1, 1, 639, 639])), (torch.Size([1, 1, 390, 3]), torch.Size([1, 1, 390, 3]), torch.Size([1, 42, 390, 390]), torch.Size([1, 1, 1, 390, 390])), (None, None, None, None)]
    >>> interpred = InterPred()

    >>> out_batch, targets = forward_batch(batch, interpred)
    >>> len(out_batch)
    2
    >>> [e.shape for e in out_batch]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    >>> [e.shape for e in targets]
    [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    >>> loss = get_loss(out_batch, targets)
    >>> loss
    tensor(..., grad_fn=<DivBackward0>)
    """
    n = len(out_batch)
    loss = 0.
    for i in range(n):
        inp = out_batch[i].float()
        target = targets[i].float()
        mask = get_mask(target)
        loss_on = torch.nn.functional.mse_loss(inp[~mask][None, ...], target[~mask][None, ...], reduction='mean')
        loss_off = torch.nn.functional.mse_loss(inp[mask][None, ...], target[mask][None, ...], reduction='mean')
        w_on = 1.
        w_off = 1.
        loss += (w_on * loss_on + w_off * loss_off) / (w_on + w_off)
    loss = loss / n
    return loss


def get_mask(intercmap, threshold=8.):
    mask = intercmap > threshold
    return mask


def get_native_contact_fraction(out_batch, targets):
    """
    # >>> dataset = PDBloader.PDBdataset('/media/bougui/scratch/pdb', randomize=False)
    # >>> dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, collate_fn=PDBloader.collate_fn)
    # >>> dataiter = iter(dataloader)
    # >>> batch = next(dataiter)
    # >>> [(A.shape, B.shape, interseq.shape, cmap.shape) if A is not None else (A, B, interseq, cmap) for A, B, interseq, cmap in batch]
    # [(None, None, None, None), (torch.Size([1, 1, 639, 3]), torch.Size([1, 1, 639, 3]), torch.Size([1, 42, 639, 639]), torch.Size([1, 1, 1, 639, 639])), (torch.Size([1, 1, 390, 3]), torch.Size([1, 1, 390, 3]), torch.Size([1, 42, 390, 390]), torch.Size([1, 1, 1, 390, 390])), (None, None, None, None)]
    # >>> interpred = InterPred()

    # >>> out_batch, targets = forward_batch(batch, interpred)
    # >>> len(out_batch)
    # 2
    # >>> [e.shape for e in out_batch]
    # [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    # >>> [e.shape for e in targets]
    # [torch.Size([1, 639, 639]), torch.Size([1, 390, 390])]
    # >>> ncf = get_native_contact_fraction(out_batch, targets)
    # >>> ncf
    # tensor(...)
    """
    n = len(out_batch)
    ncf = 0.  # Native Contacts Fraction
    with torch.no_grad():
        for i in range(n):
            inp = out_batch[i].float()
            target = targets[i].float()
            sel_contact = (target == 1)
            sel_nocontact = (target == 0)
            r_contact = (sel_contact.sum() - abs(inp[sel_contact] - target[sel_contact]).sum()) / sel_contact.sum()
            r_nocontact = (sel_nocontact.sum() -
                           abs(inp[sel_nocontact] - target[sel_nocontact]).sum()) / sel_nocontact.sum()
            ncf += (r_contact + r_nocontact) / 2.
        ncf = ncf / n
    return ncf


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
    from datetime import datetime
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
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--train', help='Train the interpred model', action='store_true')
    parser.add_argument('--nepoch', help='Number of epochs for training (default 10)', default=10, type=int)
    parser.add_argument('--batch_size', help='Batch size for training (default 4)', default=4, type=int)
    parser.add_argument('--pdbpath', help='Path to the pdb database')
    parser.add_argument('--print_each', help='Print each given steps in log file', default=100, type=int)
    parser.add_argument('--predict',
                        help='Predict for the given pdb files (see pdb1, pdb2 options)',
                        action='store_true')
    parser.add_argument('--pdb1', help='First PDB for predicrion')
    parser.add_argument('--pdb2', help='Second PDB for predicrion')
    parser.add_argument('--sel1', help='First selection', default='all')
    parser.add_argument('--sel2', help='Second selection', default='all')
    parser.add_argument('--ground_truth', help='Compute ground truth from the given pdb', action='store_true')
    parser.add_argument('--model', help='pth filename for the model to load', default='models/test.pth')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
    if args.train:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        learn(pdbpath=args.pdbpath,
              pdblist=None,
              nepoch=args.nepoch,
              batch_size=args.batch_size,
              num_workers=None,
              print_each=args.print_each,
              modelfilename=f'models/interpred_{current_time}.pth')
    if args.predict:
        intercmap = predict(pdb_a=args.pdb1,
                            pdb_b=args.pdb2,
                            sel_a=args.sel1,
                            sel_b=args.sel2,
                            modelfilename=args.model)
        if args.ground_truth:
            coords_a = utils.get_coords(args.pdb1, selection=f'polymer.protein and name CA and {args.sel1}')
            coords_b = utils.get_coords(args.pdb2, selection=f'polymer.protein and name CA and {args.sel2}')
            target = utils.get_inter_dmat(coords_a, coords_b)
            target = torch.squeeze(target.detach().cpu()).numpy()
            loss = get_loss([torch.tensor(intercmap)[None, ...]], [torch.tensor(target)[None, ...]])
            resolution = torch.sqrt(loss)
            print(f'resolution: {resolution:.4f} Å')
            fig, axs = plt.subplots(1, 2)
            _ = axs[1].matshow(target)
            _ = axs[1].set_title('Ground truth')
            _ = axs[0].matshow(intercmap)
            _ = axs[0].set_title('Prediction')
        else:
            fig, axs = plt.subplots(1, 1)
            axs.matshow(intercmap)
            axs.set_title('Prediction')
        plt.show()
