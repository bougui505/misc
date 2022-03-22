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


def sliding_mse(A, w):
    """
    >>> coords = torch.randn((10, 3))
    >>> A = get_dmat(coords)
    >>> A.shape
    torch.Size([10, 10])
    >>> coords_w = coords[2:6]
    >>> w = get_dmat(coords_w)
    >>> w.shape
    torch.Size([4, 4])
    >>> smse = sliding_mse(A, w)
    >>> ind = smse.diagonal().argmin()
    >>> ind
    tensor(2)
    >>> smse.diagonal()[ind]
    tensor(0.)
    """
    A = addbatchchannel(A)
    w = addbatchchannel(w)
    # conv = torch.nn.functional.conv2d(A, w)
    # print(conv.shape)
    A_unfold = torch.nn.functional.unfold(A, w.shape[-2:])
    # print(A_unfold.shape, w.view(w.size(0), -1).t().shape)
    out_unfold = (A_unfold - w.flatten()[None, ..., None])**2
    out_unfold = out_unfold.sum(axis=1)
    n = int(np.sqrt(out_unfold.shape[1]))
    out = out_unfold.reshape((n, n))
    return out


def torchify(x):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    try:
        x = torch.from_numpy(x)
    except TypeError:
        pass
    x = x.to(device)
    x = x.float()
    return x


def get_dmat(coords, standardize=False, mu=None, sigma=None):
    """
    >>> coords = torch.randn((10, 3))
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([10, 10])
    """
    dmat = torch.cdist(coords, coords)
    if standardize:
        dmat = standardize_dmat(dmat, mu=mu, sigma=sigma)
    return dmat


def addbatchchannel(dmat):
    """
    >>> coords = torch.randn((10, 3))
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([10, 10])
    >>> dmat = addbatchchannel(dmat)
    >>> dmat.shape
    torch.Size([1, 1, 10, 10])
    """
    if dmat.ndim == 2:
        dmat = torch.unsqueeze(dmat, 0)
        dmat = torch.unsqueeze(dmat, 1)
    return dmat


def get_cmap(coords):
    dmat = get_dmat(coords)
    dmat -= dmat.mean()
    cmap = torch.nn.functional.sigmoid(dmat)
    return cmap


def standardize_dmat(dmat, mu=None, sigma=None):
    if mu is None:
        mu = dmat.mean()
    if sigma is None:
        sigma = dmat.std()
    return (dmat - mu) / sigma


def templatematching(dmat, dmat_ref):
    """
    See: https://github.com/hirune924/TemplateMatching/blob/master/Template%20Matching%20(PyTorch%20implementation).ipynb

    >>> coords_ref = torch.randn((10, 3)) * 10.
    >>> coords = coords_ref[4:]
    >>> dmat_ref = get_dmat(coords_ref, standardize=True)
    >>> dmat = get_dmat(coords, standardize=True)

    # >>> p = plt.matshow(dmat.numpy().squeeze())
    # >>> plt.savefig('test_dmat.png')
    # >>> p = plt.matshow(dmat_ref.numpy().squeeze())
    # >>> plt.savefig('test_dmat_ref.png')

    >>> conv = templatematching(dmat, dmat_ref)

    >>> conv.shape
    torch.Size([5, 5])

    # >>> p = plt.matshow(conv.numpy().squeeze())
    # >>> plt.savefig('test_templatematching.png')
    """
    dmat = addbatchchannel(dmat)
    dmat_ref = addbatchchannel(dmat_ref)
    result = sliding_mse(dmat_ref, dmat)
    return result
    # result1 = torch.nn.functional.conv2d(dmat_ref, dmat, bias=None, stride=1, padding=0)
    # result2 = torch.sqrt(
    #     torch.sum(dmat**2) *
    #     torch.nn.functional.conv2d(dmat_ref**2, torch.ones_like(dmat), bias=None, stride=1, padding=0))
    # return (result1 / result2).squeeze(0).squeeze(0)


def get_offset(dmat, dmat_ref):
    """
    >>> coords_ref = torch.randn((10, 3)) * 10.
    >>> ind = np.random.choice(len(coords_ref) - 2)
    >>> coords = coords_ref[ind:]
    >>> dmat_ref = get_dmat(coords_ref, standardize=True)
    >>> dmat = get_dmat(coords, standardize=True)
    >>> offset, score = get_offset(dmat, dmat_ref)
    >>> offset == ind
    tensor(True)
    """
    conv = templatematching(dmat, dmat_ref)
    diag = torch.diagonal(conv, 0)
    offset = diag.argmin()
    score = diag.mean()
    return offset, score


class Mover(torch.nn.Module):
    """
    >>> A = torch.rand((10, 3))
    >>> A.shape
    torch.Size([10, 3])
    >>> mover = Mover(len(A))
    >>> B = mover(A)
    >>> B.shape
    torch.Size([10, 3])
    """
    def __init__(self, n):
        """
        n: number of points to move
        """
        super().__init__()
        self.delta = torch.nn.Parameter(torch.randn((n, 3)))

    def __call__(self, x):
        out = x + self.delta
        return out


class FlexFitter(torch.nn.Module):
    """
    Torch model (https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
    >>> A = torch.rand((10, 3))
    >>> A.shape
    torch.Size([10, 3])
    >>> ff = FlexFitter(A)
    >>> x = ff(A)
    >>> x.shape
    torch.Size([10, 3])

    >>> print(len(list(ff.parameters())))
    1

    """
    def __init__(self, coords_init):
        super(FlexFitter, self).__init__()
        self.coords_init = coords_init[:]
        self.n = len(self.coords_init)
        self.mover = Mover(self.n)

    def forward(self, x):
        x = self.mover(x)
        return x


def get_loss_dmat(dmat, dmat_ref):
    """
    >>> coords = torch.randn((10, 3))
    >>> dmat_ref = get_dmat(coords)
    >>> dmat_ref.shape
    torch.Size([10, 10])
    >>> coords_w = coords[2:6]
    >>> dmat = get_dmat(coords_w)
    >>> dmat.shape
    torch.Size([4, 4])
    >>> get_loss_dmat(dmat, dmat_ref)
    tensor(...)
    """
    # conv = templatematching(dmat, dmat_ref)
    offset, score = get_offset(dmat, dmat_ref)
    return score
    # return -torch.diagonal(conv, 0).mean()


def get_loss_rms(coords, coords_ref):
    rms = ((coords - coords_ref)**2).mean()
    return rms


def get_rmsd(A, B):
    R, t = ICP.ICP.find_rigid_alignment(A, B)
    A_fit = ICP.ICP.transform(A, R, t)
    rmsd = torch.sqrt(((B - A_fit)**2).mean())
    return rmsd


def fit(inp, target, maxiter, stop=1e-3, verbose=True, lr=0.001, save_traj=None):
    """
    >>> inp = torch.rand((8, 3))
    >>> target = torch.rand((10, 3))
    >>> output, loss, dmat_inp, dmat_ref, dmat = fit(inp, target, maxiter=10000, verbose=False)
    >>> f = plt.matshow(dmat_ref.detach().numpy())
    >>> plt.savefig('dmat_ref_test.png')
    >>> f = plt.matshow(dmat.detach().numpy())
    >>> plt.savefig('dmat_test.png')
    """
    ff = FlexFitter(inp)
    optimizer = torch.optim.Adam(ff.parameters(), lr=lr)
    dmat_ref = get_dmat(target, standardize=False)
    mu = dmat_ref.mean()
    sigma = dmat_ref.std()
    dmat_ref = get_dmat(target, standardize=False, mu=mu, sigma=sigma)
    dmat_inp = get_dmat(inp, standardize=False, mu=mu, sigma=sigma)
    if verbose:
        pbar = tqdm.tqdm(total=maxiter)
    losses = []
    loss_std_range = 100
    if save_traj is not None:
        traj = [inp.numpy()]
    rmsd = np.inf
    for i in range(maxiter):
        optimizer.zero_grad()
        output = ff(inp)
        if save_traj is not None:
            traj.append(output.detach().numpy())
        dmat = get_dmat(output, standardize=False, mu=mu, sigma=sigma)
        loss_dmat = get_loss_dmat(dmat, dmat_ref)
        loss_rms = get_loss_rms(output, inp)
        loss = loss_dmat  # + 0.001 * loss_rms
        losses.append(loss.detach())
        loss_std = np.std(losses[-loss_std_range:])
        loss.backward(retain_graph=False)
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
    return output, loss, dmat_inp, dmat_ref, dmat


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
