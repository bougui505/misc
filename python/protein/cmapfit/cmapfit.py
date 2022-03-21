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
import tqdm
import ICP
import numpy as np
import itertools
import matplotlib.pyplot as plt


def torchify(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        x = torch.from_numpy(x)
    except TypeError:
        pass
    x = x.to(device)
    x = x.float()
    return x


def get_dmat(coords):
    """
    >>> coords = torch.randn((1, 10, 3))
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([1, 1, 10, 10])
    >>> coords = torch.randn((4, 10, 3))
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([4, 1, 10, 10])
    """
    dmat = []
    for c in coords:
        dmat_ = torch.cdist(c, c)
        dmat.append(dmat_[None, ...])
    dmat = torch.vstack(tuple(dmat))
    dmat = addbatchchannel(dmat)
    return dmat


def addbatchchannel(dmat):
    """
    """
    if dmat.ndim == 2:
        dmat = torch.unsqueeze(dmat, 0)
        dmat = torch.unsqueeze(dmat, 1)
    if dmat.ndim == 3:
        dmat = torch.unsqueeze(dmat, 1)
    return dmat


def split_coords(coords, split_size):
    """
    >>> coords = torch.randn((17, 3))
    >>> batch = split_coords(coords, 4)
    >>> batch.shape
    torch.Size([4, 4, 3])
    >>> coords = torch.randn((16, 3))
    >>> batch = split_coords(coords, 4)
    >>> batch.shape
    torch.Size([4, 4, 3])

    If split_size is larger than the number of coords return one batch:

    >>> coords = torch.randn((16, 3))
    >>> batch = split_coords(coords, 17)
    >>> batch.shape
    torch.Size([1, 16, 3])
    """
    coords = torch.squeeze(coords)
    batch = torch.split(coords, split_size)
    if batch[-1].shape != batch[0].shape:
        batch = batch[:-1]
    batch = torch.stack(batch)
    return batch


def templatematching(dmat, dmat_ref):
    """
    See: https://github.com/hirune924/TemplateMatching/blob/master/Template%20Matching%20(PyTorch%20implementation).ipynb

    >>> coords_ref = torch.randn((1, 17, 3)) * 10.
    >>> coords = split_coords(coords_ref, 4)
    >>> coords.shape
    torch.Size([4, 4, 3])
    >>> dmat_ref = get_dmat(coords_ref)
    >>> dmat_ref.shape
    torch.Size([1, 1, 17, 17])
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([4, 1, 4, 4])
    >>> conv = templatematching(dmat, dmat_ref)

    >>> conv.shape
    torch.Size([4, 14, 14])
    """
    # dmat = addbatchchannel(dmat)
    # dmat_ref = addbatchchannel(dmat_ref)
    result1 = torch.nn.functional.conv2d(dmat_ref, dmat, bias=None, stride=1, padding=0)
    result2 = torch.sqrt(
        torch.sum(dmat**2) *
        torch.nn.functional.conv2d(dmat_ref**2, torch.ones_like(dmat), bias=None, stride=1, padding=0))
    return (result1 / result2).squeeze(0).squeeze(0)


def get_offset(dmat, dmat_ref):
    """
    >>> coords_ref = torch.randn((1, 17, 3)) * 10.
    >>> coords = split_coords(coords_ref + torch.randn((1, 17, 3)), 4)
    >>> coords.shape
    torch.Size([4, 4, 3])
    >>> dmat_ref = get_dmat(coords_ref)
    >>> dmat_ref.shape
    torch.Size([1, 1, 17, 17])
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([4, 1, 4, 4])
    >>> offset = get_offset(dmat, dmat_ref)
    >>> offset.shape
    torch.Size([4])
    >>> offset
    tensor([ 0,  4,  8, 12])
    """
    conv = templatematching(dmat, dmat_ref)
    if conv.ndim == 2:
        diag = torch.diagonal(conv)[None, ...]
    else:
        diag = torch.diagonal(conv, dim1=1, dim2=2)
    # print(diag.shape)
    offset = diag.argmax(dim=1)
    return offset


class Mover(torch.nn.Module):
    """
    The first dimension is a batch 
    >>> n = 10
    >>> batchsize = 2
    >>> A = torch.rand((batchsize, n, 3))
    >>> A.shape
    torch.Size([2, 10, 3])
    >>> mover = Mover(n, batchsize)
    >>> B = mover(A)
    >>> B.shape
    torch.Size([2, 10, 3])
    """
    def __init__(self, n, batchsize):
        """
        n: number of points to move
        """
        super().__init__()
        self.delta = torch.nn.Parameter(torch.randn((batchsize, n, 3)))

    def __call__(self, x):
        out = x + self.delta
        return out


class FlexFitter(torch.nn.Module):
    """
    Torch model (https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
    >>> A = torch.rand((2, 10, 3))
    >>> A.shape
    torch.Size([2, 10, 3])
    >>> ff = FlexFitter(A)
    >>> x = ff(A)
    >>> x.shape
    torch.Size([2, 10, 3])

    >>> print(len(list(ff.parameters())))
    1

    """
    def __init__(self, coords_init):
        super(FlexFitter, self).__init__()
        self.coords_init = coords_init[:]
        self.n = self.coords_init.shape[1]
        self.batchsize = self.coords_init.shape[0]
        self.mover = Mover(self.n, self.batchsize)

    def forward(self, x):
        x = self.mover(x)
        return x


def get_loss_dmat(dmat, dmat_ref):
    """
    >>> coords_ref = torch.randn((1, 31, 3)) * 10.
    >>> coords = split_coords(coords_ref, 5)
    >>> coords.shape
    torch.Size([6, 5, 3])
    >>> dmat_ref = get_dmat(coords_ref)
    >>> dmat_ref.shape
    torch.Size([1, 1, 31, 31])
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([6, 1, 5, 5])
    >>> loss = get_loss_dmat(dmat, dmat_ref)
    >>> loss < 1e-5
    tensor(True)
    """
    # conv = templatematching(dmat, dmat_ref)
    offsets = get_offset(dmat, dmat_ref)
    n = dmat.shape[2]
    loss = 0
    nbatch = len(offsets)
    for i, offset in enumerate(offsets):
        dmat_ref_aln = torch.squeeze(dmat_ref)[offset:, offset:]
        dmat_ref_aln = dmat_ref_aln[:n, :n]
        loss += ((dmat[i, 0, ...] - dmat_ref_aln)**2).mean()
    loss /= nbatch
    return loss


def get_loss_rms(coords, coords_ref):
    """
    >>> coords_ref = torch.randn((1, 30, 3)) * 10.
    >>> coords_ref.shape
    torch.Size([1, 30, 3])
    >>> coords = split_coords(coords_ref, 5)
    >>> coords.shape
    torch.Size([6, 5, 3])
    >>> get_loss_rms(coords, coords_ref)
    tensor(0.)
    """
    nbatch = coords.shape[0]
    n = coords.shape[1]
    coords = coords.reshape(nbatch * n, 3)
    nbatch_ref = coords_ref.shape[0]
    n_ref = coords_ref.shape[1]
    coords_ref = coords_ref.reshape(nbatch_ref * n_ref, 3)
    assert coords.shape == coords_ref.shape
    rms = ((coords - coords_ref)**2).mean()
    return rms


def unbatch_coords(coords):
    out = coords
    nbatch, n, _ = out.shape
    return out.reshape(nbatch * n, 3)


def unbatch_dmat(coords):
    coords = unbatch_coords(coords)[None, ...]
    dmat = torch.squeeze(get_dmat(coords))
    return dmat


def fit(inp, target, maxiter, split_size, stop=1e-4, verbose=True, lr=0.01, save_traj=None):
    """
    >>> coords_ref = torch.randn((1, 132, 3)) * 10.
    >>> coords = torch.randn((1, 132, 3)) * 10.
    >>> output, loss, dmat_inp, dmat_ref, dmat = fit(coords, coords_ref, split_size=4, maxiter=100, verbose=True, stop=-np.inf)
    >>> output.shape
    torch.Size([132, 3])
    >>> dmat_inp.shape
    torch.Size([132, 132])
    >>> dmat_ref.shape
    torch.Size([132, 132])
    >>> dmat.shape
    torch.Size([132, 132])

    # >>> f = plt.matshow(dmat_ref.detach().numpy())
    # >>> plt.savefig('dmat_ref_test.png')
    # >>> f = plt.matshow(dmat.detach().numpy())
    # >>> plt.savefig('dmat_test.png')
    """
    if inp.ndim == 2:
        # Add a batch dimanesion
        inp = torch.unsqueeze(inp, 0)
    if target.ndim == 2:
        target = torch.unsqueeze(target, 0)
    inp = split_coords(inp, split_size=split_size)
    ff = FlexFitter(inp)
    optimizer = torch.optim.Adam(ff.parameters(), lr=lr)
    dmat_ref = get_dmat(target)
    # dmat_inp = get_dmat(inp)
    if verbose:
        pbar = tqdm.tqdm(total=maxiter)
    losses = []
    loss_std_range = 100
    if save_traj is not None:
        traj = [unbatch_coords(inp).numpy()]
    rmsd = np.inf
    for i in range(maxiter):
        optimizer.zero_grad()
        output = ff(inp)
        if save_traj is not None:
            traj.append(unbatch_coords(output).detach().numpy())
        dmat = get_dmat(output)
        loss_dmat = get_loss_dmat(dmat, dmat_ref)
        loss_rms = get_loss_rms(output, inp)
        loss = loss_dmat  # + 0.001 * loss_rms
        losses.append(loss.detach())
        loss_std = np.std(losses[-loss_std_range:])
        loss.backward(retain_graph=True)
        optimizer.step()
        rmsdmat = np.sqrt(loss_dmat.detach().numpy())
        rmsd_prev = rmsd
        rmsd = np.sqrt(loss_rms.detach().numpy())
        delta_rmsd = rmsd - rmsd_prev
        if verbose:
            # pbar.set_description(desc=f'loss: {loss:.3f}; RMSD: {rmsd:.3f}')
            pbar.set_description(
                desc=
                f'loss: {loss:.3f}±{loss_std:.3e}; rmsd: {rmsd:.3f}; rmsdmat: {rmsdmat:.3f}; deltarmsd: {delta_rmsd:.3e}'
            )
            pbar.update(1)
        if np.abs(delta_rmsd) <= stop:
            if verbose:
                print(f"Early stop at loss: {loss:.3f} ± {loss_std:.3e} with deltarmsd: {delta_rmsd:.3e}/{stop}")
            break
    if save_traj is not None:
        traj = np.asarray(traj)
        print(f'Trajectory shape: {traj.shape}')
        np.save(save_traj, traj)
    return unbatch_coords(output), loss, unbatch_dmat(inp), unbatch_dmat(target), unbatch_dmat(output)


if __name__ == '__main__':
    from pymol import cmd
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--pdb1')
    parser.add_argument('--pdb2')
    parser.add_argument('-n', '--maxiter', help='Maximum number of minimizer iterations', default=5000, type=int)
    parser.add_argument('--splitsize',
                        help='Size of the fragment to fit (default: no fragment)',
                        type=int,
                        default=1000000)
    parser.add_argument('--lr', help='Learning rate for the optimizer (Adam) -- default=0.01', default=0.01, type=float)
    parser.add_argument('--save_traj', help='Save the trajectory minimization in the given npy file')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cmd.load(args.pdb1, 'pdb1')
    cmd.load(args.pdb2, 'pdb2')
    pdb1 = torchify(cmd.get_coords('pdb1 and polymer.protein and name CA'))
    pdb2 = torchify(cmd.get_coords('pdb2 and polymer.protein and name CA'))
    coordsfit, loss, dmat_inp, dmat_ref, dmat_out = fit(pdb1,
                                                        pdb2,
                                                        args.maxiter,
                                                        split_size=args.splitsize,
                                                        stop=1e-4,
                                                        verbose=True,
                                                        lr=args.lr,
                                                        save_traj=args.save_traj)
    plt.matshow(dmat_inp.detach().numpy())
    plt.savefig('dmat_inp.png')
    plt.matshow(dmat_out.detach().numpy())
    plt.savefig('dmat_out.png')
    plt.matshow(dmat_ref.detach().numpy())
    plt.savefig('dmat_ref.png')
