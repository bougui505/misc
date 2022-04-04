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
import numpy as np
import matplotlib.pyplot as plt
from misc.localpeaks import Local_peaks
from misc.pytorch import PDBloader


def sliding_mse(A, w, padding=0, diagonal=False):
    """
    Sub region matching
    >>> coords = torch.randn((10, 3))
    >>> A = get_dmat(coords)
    >>> A.shape
    torch.Size([10, 10])
    >>> coords_w = coords[2:6]
    >>> w = get_dmat(coords_w)
    >>> w.shape
    torch.Size([4, 4])
    >>> smse = sliding_mse(A, w)
    >>> smse.shape
    torch.Size([7, 7])
    >>> ind = smse.diagonal().argmin()
    >>> ind
    tensor(2)
    >>> smse.diagonal()[ind]
    tensor(0.)

    >>> smse = sliding_mse(A, w, padding='full')
    >>> smse.shape
    torch.Size([15, 15])
    >>> ind_pad = smse.diagonal().argmin()
    >>> ind_pad == w.shape[0] + ind
    tensor(True)

    # With odd kernel shape:
    >>> coords_w = coords[2:7]
    >>> w = get_dmat(coords_w)
    >>> w.shape
    torch.Size([5, 5])
    >>> smse = sliding_mse(A, w, padding='full')
    >>> smse.shape
    torch.Size([16, 16])
    >>> ind_pad = smse.diagonal().argmin()
    >>> ind_pad == w.shape[0] + ind
    tensor(True)

    # Compute the convolution only on the diagonal
    >>> coords = torch.randn((50, 3))
    >>> A = get_dmat(coords)
    >>> coords_w = coords[7:17]
    >>> w = get_dmat(coords_w)
    >>> w.shape
    torch.Size([10, 10])
    >>> smse = sliding_mse(A, w, diagonal=True)
    >>> smse.shape
    torch.Size([1, 40])
    >>> ind = smse.argmin()
    >>> ind
    tensor(7)

    # Compute the convolution only on the diagonal with full padding:
    >>> smse = sliding_mse(A, w, diagonal=True, padding='full')
    >>> smse.shape
    torch.Size([1, 60])
    >>> ind_pad = smse.argmin()
    >>> ind_pad == ind + w.shape[0]
    tensor(True)
    """
    A = addbatchchannel(A)
    w = addbatchchannel(w)
    w = torchify(w)
    N, C, H, W = A.shape
    # conv = torch.nn.functional.conv2d(A, w)
    # print(conv.shape)
    if not diagonal:
        if padding == 'full':
            padding = w.shape[-2:]
        A_unfold = torch.nn.functional.unfold(A, w.shape[-2:], padding=padding)
    else:  # diagonal
        A_unfold = unfold_diagonal(A, w.shape[-2:], padding=padding)
    A_unfold = torchify(A_unfold)
    out_unfold = (A_unfold - w.flatten()[None, ..., None])**2
    out_unfold = out_unfold.mean(axis=1)
    if diagonal:
        return out_unfold
    else:
        n = int(np.sqrt(out_unfold.shape[1]))
        out = out_unfold.reshape((n, n))
        return out


def unfold_diagonal(A, kernel_size, padding=0):
    """
    >>> A = torch.randn((3, 2, 50, 50))
    >>> A_unfold = unfold_diagonal(A, (10, 10))
    >>> A_unfold.shape
    torch.Size([3, 200, 40])

    # Padding:
    >>> A_unfold = unfold_diagonal(A, (10, 10), padding=3)
    >>> A_unfold.shape
    torch.Size([3, 200, 46])
    >>> A_unfold = unfold_diagonal(A, (10, 10), padding='full')
    >>> A_unfold.shape
    torch.Size([3, 200, 60])
    """
    if padding == 'full':
        padding = tuple(kernel_size) + tuple(kernel_size)
    else:
        padding = (padding, ) * 4
    A = torch.nn.functional.pad(A, padding)
    N, C, H, W = A.shape
    assert H == W, f"A is not symmetric: {A.shape}"  # Assert a square input tensor
    h, w = kernel_size
    A_unfold = torch.zeros((N, C * h * w, H - h))
    for i in range(H - h):
        A_ = A[:, :, i:i + h, i:i + w].reshape((N, C * h * w))
        A_unfold[:, :, i] = A_
    return A_unfold


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


def get_cmap(dmat, threshold=8.):
    """
    >>> dmat = torch.arange(10)
    >>> dmat
    tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    >>> get_cmap(dmat)
    tensor([1.0000, 0.8750, 0.7500, 0.6250, 0.5000, 0.3750, 0.2500, 0.1250, -0.0000,
            0.0000])
    """
    cmap = dmat - threshold
    cmap = torch.nn.functional.relu(-cmap)
    cmap = cmap / threshold
    # cmap = threshold - cmap
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
    score = diag.min()
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
        randw = torchify(torch.randn((n, 3)))
        self.delta = torch.nn.Parameter(randw)

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


def get_P(n, nout):
    P = torch.eye(n)
    P = P[:nout, ...]
    return P


def permute(x, P):
    x = torch.matmul(P, x)
    x = torch.matmul(x, P.T)
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


def get_loss_permutation(P):
    eps = 1e-3
    loss_1 = (P * torch.log(P + eps)).mean()
    loss_2 = ((1. - P.sum(axis=0))**2).mean() + ((1. - P.sum(axis=1))**2).mean()
    loss = loss_1 + loss_2
    return loss


def fit(inp, target, maxiter, stop=1e-3, verbose=True, lr=0.001, save_traj=None, contact=False):
    """
    >>> inp = torch.rand((8, 3))
    >>> target = torch.rand((10, 3))
    >>> output, loss, dmat_inp, dmat_ref, dmat = fit(inp, target, maxiter=10000, lr=0.01, stop=1e-8, verbose=False)

    # >>> f = plt.matshow(dmat_ref.detach().numpy())
    # >>> plt.savefig('dmat_ref_test.png')
    # >>> f = plt.matshow(dmat.detach().numpy())
    # >>> plt.savefig('dmat_test.png')
    """
    # torch.autograd.set_detect_anomaly(True)
    logging.info('Fitting')
    ff = FlexFitter(inp)
    dmat_ref = get_dmat(target, standardize=False)
    mu = dmat_ref.mean()
    sigma = dmat_ref.std()
    dmat_ref = get_dmat(target, standardize=False, mu=mu, sigma=sigma)
    if contact:
        dmat_ref = get_cmap(dmat_ref)
    dmat_inp = get_dmat(inp, standardize=False, mu=mu, sigma=sigma)
    if contact:
        dmat_inp = get_cmap(dmat_inp)
    # autocropper = Autocrop(dmat_ref, dmat_inp.shape[-1])
    optimizer = torch.optim.Adam(ff.parameters(), lr=lr)
    # optimizer_cropper = torch.optim.Adam(autocropper.parameters(), lr=0.1)
    if verbose:
        pbar = tqdm.tqdm(total=maxiter)
    losses = []
    loss_std_range = 100
    if save_traj is not None:
        traj = [inp.cpu().numpy()]
    rmsd = np.inf
    # P = get_P(dmat_ref.shape[-1], dmat_inp.shape[-1])
    for i in range(maxiter):
        optimizer.zero_grad()
        # optimizer_cropper.zero_grad()
        output = ff(inp)
        if save_traj is not None:
            traj.append(output.detach().cpu().numpy())
        dmat = get_dmat(output, standardize=False, mu=mu, sigma=sigma)
        if contact:
            dmat = get_cmap(dmat)
        # P_out = autocropper(P)
        # dmat_ref_crop = permute(dmat_ref, P_out)
        loss_dmat = get_loss_dmat(dmat, dmat_ref)
        loss_dmat_auto = get_loss_dmat(dmat, dmat_inp)
        loss_rms = get_loss_rms(output, inp)
        # loss_P = get_loss_permutation(P_out)
        loss = loss_dmat
        losses.append(loss.detach().cpu())
        loss_std = np.std(losses[-loss_std_range:])
        loss.backward(retain_graph=False)
        optimizer.step()
        # optimizer_cropper.step()
        # logging.info(autocropper.P.mean())
        rmsdmat_target = np.sqrt(loss_dmat.detach().cpu().numpy())
        rmsdmat_inp = np.sqrt(loss_dmat_auto.detach().cpu().numpy())
        rmsd_prev = rmsd
        rmsd = np.sqrt(loss_rms.detach().cpu().numpy())
        delta_rmsd = rmsd - rmsd_prev
        if i == 0:
            logging.info(f'Initial_loss: {loss:.3f}')
            logging.info(f'Initial_rmsd: {rmsd:.3f}')
            logging.info(f'Initial_rmsdmat: {rmsdmat_target:.3f}')
            logging.info(f'Initial_rmsdmat_inp: {rmsdmat_inp:.3f}')
        if verbose:
            # pbar.set_description(desc=f'loss: {loss:.3f}; RMSD: {rmsd:.3f}')
            pbar.set_description(
                desc=
                f'loss: {loss:.3f}; rmsd: {rmsd:.3f}; rmsdmat_target: {rmsdmat_target:.3f}; rmsdmat_inp: {rmsdmat_inp:.3f}; deltarmsd: {delta_rmsd:.3e}'
            )
            pbar.update(1)
        if np.abs(delta_rmsd) <= stop:
            if verbose:
                print(f"Early stop at loss: {loss:.3f} ± {loss_std:.3e} with deltarmsd: {delta_rmsd:.3e}/{stop}")
            break
    logging.info(f'Final_loss: {loss:.3f}')
    logging.info(f'Final_rmsd: {rmsd:.3f}')
    logging.info(f'Final_rmsdmat: {rmsdmat_target:.3f}')
    logging.info(f'Final_rmsdmat_inp: {rmsdmat_inp:.3f}')
    if save_traj is not None:
        traj = np.asarray(traj)
        print(f'Trajectory shape: {traj.shape}')
        np.save(save_traj, traj)
    return output, loss, dmat_inp, dmat_ref, dmat


def generate_trace():
    """
    >>> coords = generate_trace()
    >>> coords.shape
    torch.Size([259, 3])
    >>> torch.norm(coords[1:] - coords[:-1], dim=1).mean()
    tensor(3.7984)
    """
    cmd.load('data/6i3r_A.pdb')
    coords = cmd.get_coords('polymer.protein and name CA')
    # coords = [np.random.normal(size=3)]
    # while len(coords) < n:
    #     v = np.random.normal(size=3)
    #     v /= np.linalg.norm(v)
    #     v *= 3.8
    #     c = coords[-1] + v
    #     cdist = scidist.cdist(c[None, :], np.asarray(coords))
    #     dmin = cdist.min()
    #     dmax = cdist.max()
    #     if dmin >= 3.8 and dmax < 50:
    #         coords.append(c)
    # coords = np.asarray(coords)
    # # coords -= coords.mean(axis=0)
    return torch.Tensor(coords)


class Profile(object):
    def __init__(self, dmat, dmat_ref, coords=None, coords_ref=None):
        """
        # First example with dmat.shape < dmat_ref.shape
        >>> coords = generate_trace()
        >>> dmat_ref = get_dmat(coords)
        >>> coords_w = coords[7:100]
        >>> dmat = get_dmat(coords_w)
        >>> dmat.shape
        torch.Size([93, 93])
        >>> profile = Profile(dmat, dmat_ref, coords=coords_w, coords_ref=coords)
        >>> profile.argmin()
        7
        >>> profile.localminima
        array([7])
        >>> profile.plot(filename='profile_test1.png')
        >>> dmat_aln, dmat_ref_aln = profile.map_aligned()
        >>> dmat.shape, dmat_ref.shape
        (torch.Size([93, 93]), torch.Size([259, 259]))
        >>> [d.shape for d in dmat_aln], [d.shape for d in dmat_ref_aln]
        ([torch.Size([93, 93])], [torch.Size([93, 93])])
        >>> torch.isclose(((get_cmap(dmat_aln[0]) - get_cmap(dmat_ref_aln[0]))**2).mean(), profile.profile.min())
        tensor(True)
        >>> coords1, coords2 = profile.split_coords()
        >>> [c.shape for c in coords1], [c.shape for c in coords2]
        ([torch.Size([93, 3])], [torch.Size([93, 3])])
        >>> [(get_dmat(c)==d).all() for c, d in zip(coords1, dmat_aln)]
        [tensor(True)]

        # Example with dmat.shape > dmat_ref.shape
        >>> coords = generate_trace()
        >>> dmat = get_dmat(coords)
        >>> coords_w = coords[7:100]
        >>> dmat_ref = get_dmat(coords_w)
        >>> profile = Profile(dmat, dmat_ref, coords, coords_w)
        >>> profile.argmin()
        7
        >>> profile.localminima
        array([7])
        >>> profile.plot(filename='profile_test2.png')
        >>> dmat_aln, dmat_ref_aln = profile.map_aligned()
        >>> dmat.shape, dmat_ref.shape
        (torch.Size([259, 259]), torch.Size([93, 93]))
        >>> [d.shape for d in dmat_aln], [d.shape for d in dmat_ref_aln]
        ([torch.Size([93, 93])], [torch.Size([93, 93])])
        >>> torch.isclose(((get_cmap(dmat_aln[0]) - get_cmap(dmat_ref_aln[0]))**2).mean(), profile.profile.min())
        tensor(True)
        >>> coords1, coords2 = profile.split_coords()
        >>> [c.shape for c in coords1], [c.shape for c in coords2]
        ([torch.Size([93, 3])], [torch.Size([93, 3])])
        >>> [(get_dmat(c)==d).all() for c, d in zip(coords1, dmat_aln)]
        [tensor(True)]

        # Example with non-contiguous domains
        >>> coords = generate_trace()
        >>> dmat_ref = get_dmat(coords[:245])
        >>> dmat_ref.shape
        torch.Size([245, 245])
        >>> coords_w = coords[np.r_[10:50, 160:250]]
        >>> dmat = get_dmat(coords_w)
        >>> dmat.shape
        torch.Size([130, 130])
        >>> profile = Profile(dmat, dmat_ref, coords_w, coords[:245])
        >>> profile.plot(filename='profile_test3.png')
        >>> profile.localminima
        array([ 10, 120])
        >>> dmat.shape, dmat_ref.shape
        (torch.Size([130, 130]), torch.Size([245, 245]))
        >>> dmat_aln, dmat_ref_aln = profile.map_aligned()
        >>> [d.shape for d in dmat_aln], [d.shape for d in dmat_ref_aln]
        ([torch.Size([130, 130]), torch.Size([125, 125])], [torch.Size([130, 130]), torch.Size([125, 125])])
        >>> coords1, coords2 = profile.split_coords()
        >>> [c.shape for c in coords1], [c.shape for c in coords2]
        ([torch.Size([130, 3]), torch.Size([125, 3])], [torch.Size([130, 3]), torch.Size([125, 3])])
        """
        self.dmat = dmat
        self.dmat_ref = dmat_ref
        self.coords = coords
        self.coords_ref = coords_ref
        if self.coords is not None:
            assert len(self.coords) == self.dmat.shape[-1]
        if self.coords_ref is not None:
            assert len(self.coords_ref) == self.dmat_ref.shape[-1]
        self.get_profile()
        self.localminima = self.get_localminima()
        self.scores, self.score, self.score_min = self.get_score()
        logging.info(f'dmat_score_mean: {self.score:.3f}')
        logging.info(f'dmat_score_min: {self.score_min:.3f}')
        scores_repr = ', '.join([f'{e:.3f}' for e in self.scores])
        logging.info(f'dmat_scores: {scores_repr}')

    @property
    def dmat1(self):
        if self.dmat.shape[-1] <= self.dmat_ref.shape[-1]:
            self.reverse_ref = False
            return self.dmat
        else:
            self.reverse_ref = True
            return self.dmat_ref

    @property
    def coords1(self):
        if self.dmat.shape[-1] <= self.dmat_ref.shape[-1]:
            self.reverse_ref = False
            return self.coords
        else:
            self.reverse_ref = True
            return self.coords_ref

    @property
    def dmat2(self):
        if self.dmat.shape[-1] <= self.dmat_ref.shape[-1]:
            self.reverse_ref = False
            return self.dmat_ref
        else:
            self.reverse_ref = True
            return self.dmat

    @property
    def coords2(self):
        if self.dmat.shape[-1] <= self.dmat_ref.shape[-1]:
            self.reverse_ref = False
            return self.coords_ref
        else:
            self.reverse_ref = True
            return self.coords

    def get_profile(self):
        self.profile = torch.squeeze(
            sliding_mse(get_cmap(self.dmat2), get_cmap(self.dmat1), diagonal=True, padding='full'))
        self.indices = np.arange(len(self.profile)) - self.dmat1.shape[0]
        return self.indices, self.profile

    def get_localminima(self):
        lm_raw = Local_peaks(self.profile.cpu().numpy(), zscore=2.5, wlen=30, minima=True, logging=logging).peaks
        # lm_raw, _ = scipy.signal.find_peaks(-self.profile.cpu().numpy(), prominence=0.05, wlen=10)
        # lm_raw = scipy.signal.argrelextrema(self.profile.cpu().numpy(), np.less)
        lm = self.indices[lm_raw]
        logging.info(f'local_minima: {lm}')
        return lm

    def plot(self, filename='profile.png'):
        plt.figure()
        plt.plot(self.indices, self.profile)
        plt.axvline(0, color='red', linestyle='-', linewidth=1.)
        plt.axvline(self.dmat2.shape[-1] - self.dmat1.shape[-1], color='gray', linestyle='-', linewidth=1.)
        for lm in self.localminima:
            plt.axvline(lm, color='blue', linestyle='--', linewidth=1.)
        plt.savefig(filename)

    def plot_dmat(self):
        dmats1, dmats2 = self.map_aligned()
        n = len(dmats1)
        plt.figure()
        plt.matshow(self.dmat1)
        plt.savefig('dmat1.png')
        plt.figure()
        plt.matshow(self.dmat2)
        plt.savefig('dmat2.png')
        for i in range(n):
            plt.figure()
            plt.matshow(dmats1[i])
            plt.savefig(f'dmats1_{i}.png')
            plt.figure()
            plt.matshow(dmats2[i])
            plt.savefig(f'dmats2_{i}.png')

    def argmin(self):
        return self.indices[self.profile.argmin()]

    def map_aligned(self):
        inds = self.localminima
        n = self.dmat1.shape[-1]
        dmats2 = []
        for ind in inds:
            dmats2.append(self.dmat2[ind:ind + n, :][:, ind:ind + n])
        dmats1 = []
        for dmat2 in dmats2:
            n = dmat2.shape[-1]
            dmat1 = self.dmat1[:n, :][:, :n]
            dmats1.append(dmat1)
        return dmats1, dmats2

    def split_coords(self):
        inds = self.localminima
        n = self.coords1.shape[0]
        coords2 = []
        for ind in inds:
            coords2.append(self.coords2[ind:ind + n, :])
        coords1 = []
        for c2 in coords2:
            n = c2.shape[0]
            c1 = self.coords1[:n, :][:, :n]
            coords1.append(c1)
        return coords1, coords2

    def get_score(self, dthreshold=8.):
        dmats1, dmats2 = self.map_aligned()
        scores = []
        for dmat1, dmat2 in zip(dmats1, dmats2):
            sel = dmat2 < dthreshold
            s = np.sqrt(((dmat1.cpu() - dmat2.cpu())**2)[sel].mean())
            scores.append(s)
        if len(scores) > 0:
            return scores, np.nanmean(scores), np.nanmin(scores)
        else:
            return scores, np.inf, np.inf


if __name__ == '__main__':
    from pymol import cmd
    import sys
    import doctest
    import argparse
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb1', required=True)
    parser.add_argument('--pdb2')
    parser.add_argument('--db')
    parser.add_argument('--fit', help='Flexible fit of contact maps', action='store_true')
    parser.add_argument('--contacts', help='Flexible fit to optimize contacts and not distances', action='store_true')
    parser.add_argument('-n', '--maxiter', help='Maximum number of minimizer iterations', default=5000, type=int)
    parser.add_argument('--lr', help='Learning rate for the optimizer (Adam) -- default=0.01', default=0.01, type=float)
    parser.add_argument('--save_traj', help='Save the trajectory minimization in the given npy file')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f'pdb1: {args.pdb1}')
    cmd.load(args.pdb1, 'pdb1')
    pdb1 = torchify(cmd.get_coords('pdb1 and polymer.protein and name CA'))
    logging.info(f'pdb1.shape: {pdb1.shape[0]}')
    dmat = get_dmat(pdb1)
    if args.pdb2 is not None:
        logging.info(f'pdb2: {args.pdb2}')
        cmd.load(args.pdb2, 'pdb2')
        pdb2 = torchify(cmd.get_coords('pdb2 and polymer.protein and name CA'))
        logging.info(f'pdb2.shape: {pdb2.shape[0]}')
        dmat_ref = get_dmat(pdb2)
        profile = Profile(dmat, dmat_ref, coords=pdb1, coords_ref=pdb2)
        print(f'{args.pdb1}|{args.pdb2}: {profile.score:.3f} Å')
    if args.db is not None:
        dataset = PDBloader.PDBdataset(args.db, return_name=True, selection='polymer.protein and name CA')
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=1,
                                                 shuffle=False,
                                                 num_workers=os.cpu_count(),
                                                 collate_fn=PDBloader.collate_fn)
        for batch in dataloader:
            name, coords = batch[0]
            if coords is not None:
                logging.info(f'pdb2: {name}')
                logging.info(f'pdb2.shape: {coords.shape[0]}')
                coords = torchify(coords)
                dmat_ref = get_dmat(coords)
                profile = Profile(dmat, dmat_ref)

    # profile.plot()
    # profile.plot_dmat()
    if args.fit:
        coords1, coords2 = profile.split_coords()
        for i, (c1, c2) in enumerate(zip(coords1, coords2)):
            out_traj = f'{os.path.splitext(args.save_traj)[0]}_{i}.npy'
            coordsfit, loss, dmat_inp, dmat_ref, dmat_out = fit(c1,
                                                                c2,
                                                                args.maxiter,
                                                                stop=1e-6,
                                                                verbose=True,
                                                                lr=args.lr,
                                                                save_traj=out_traj,
                                                                contact=args.contacts)
            plt.matshow(dmat_inp.detach().cpu().numpy())
            plt.savefig(f'dmat_inp_{i}.png')
            plt.matshow(dmat_out.detach().cpu().numpy())
            plt.savefig(f'dmat_out_{i}.png')
            plt.matshow(dmat_ref.detach().cpu().numpy())
            plt.savefig(f'dmat_ref_{i}.png')
