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
from pymol import cmd
import numpy as np
import itertools
import os


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


SCRIPTDIR = GetScriptDir()


def fetch(code, name, pdbpath=f'{SCRIPTDIR}/pdb/'):
    """
    >>> fetch(code='1ycr', name='p1')
    >>> cmd.get_coords('p1').shape
    (818, 3)
    >>> cmd.delete('all')
    >>> fetch(code='1ycr_A', name='p1')
    >>> cmd.get_coords('p1').shape
    (705, 3)
    >>> cmd.delete('all')
    >>> fetch(code='1ycrA', name='p1')
    >>> cmd.get_coords('p1').shape
    (705, 3)
    >>> cmd.delete('all')
    """
    if len(code) > 4:
        if code[4] == '_':
            chain = code[5]  # e.g. 1ycr_A
        else:
            chain = code[4]  # e.g. 1ycrA
    else:
        chain = None
    pdbfilepath = f'{pdbpath}/{code[1]}{code[2]}/pdb{code[:4]}.ent.gz'
    # if os.path.exists(pdbfilepath):
    cmd.load(pdbfilepath, object=name)
    if chain is not None:
        cmd.remove(f'{name} and not chain {chain}')


def get_coords(pdb, sel='all'):
    cmd.delete('all')
    cmd.set('fetch_path', cmd.exp_path('~/pdb'), quiet=1)
    pymolname = randomgen.randomstring()
    if os.path.exists(pdb):
        cmd.load(pdb, pymolname)
    else:  # Try to fetch from the PDB
        fetch(pdb, name=pymolname)
    coords = cmd.get_coords(selection=f'{pymolname} and polymer.protein and name CA and {sel}')
    cmd.delete(pymolname)
    coords = torch.tensor(coords)
    return coords


def get_rmsd(c1, c2):
    """
    >>> c1 = np.random.uniform(size=(10,3))
    >>> c2 = np.random.uniform(size=(10,3))
    >>> get_rmsd(c1, c2)
    0...
    """
    rmsd = np.sqrt(((c1 - c2)**2).sum(axis=1).mean())
    return rmsd


def get_gdt(c1, c2, thresholds=[1, 2, 4, 8]):
    """
    >>> c1 = np.random.uniform(size=(10,3))
    >>> c2 = np.random.uniform(size=(10,3)) + 1.
    >>> get_gdt(c1, c2)
    0...
    """
    n = len(c1)
    distances = np.linalg.norm(c2 - c1, axis=1)
    gdt = []
    for thr in thresholds:
        gdt.append((distances <= thr).sum() / n)
    return np.mean(gdt)


def get_dmat(coords):
    """
    >>> coords = torch.randn(1, 10, 3)
    >>> dmat = get_dmat(coords)
    >>> dmat.shape
    torch.Size([1, 1, 10, 10])
    """
    dmat = torch.cdist(coords, coords)
    dmat = dmat[:, None, ...]  # Add the channel dimension
    return dmat


def get_cmap(dmat, threshold=8.):
    """
    >>> from pymol import cmd
    >>> cmd.load('pdb/yc/pdb1ycr.ent.gz', 'inp')
    >>> coords = cmd.get_coords('inp and polymer.protein and name CA and chain A')
    >>> coords = torch.tensor(coords[None, ...])
    >>> dmat = get_dmat(coords)
    >>> cmap = get_cmap(dmat)
    >>> cmap.shape
    torch.Size([1, 1, 85, 85])
    >>> cmap = cmap.cpu().numpy().squeeze()

    # >>> _ = plt.matshow(cmap)
    # >>> plt.colorbar()
    # >>> plt.show()
    """
    cmap = 1. - torch.nn.functional.sigmoid(dmat - threshold)
    return cmap


def get_Rg(coords):
    """
    >>> coords = torch.randn(1, 10, 3)
    >>> Rg = get_Rg(coords)
    >>> Rg
    tensor(0...)
    """
    rk = torch.linalg.norm(coords, dim=-1)
    rmean = rk.mean()
    Rg = torch.sqrt(((rk - rmean)**2).mean())
    return Rg


def get_random_fragment(coords, fragment_ratio=None, noise=None):
    """
    >>> from pymol import cmd
    >>> cmd.load('pdb/yc/pdb1ycr.ent.gz', 'inp')
    >>> coords = cmd.get_coords('inp and polymer.protein and name CA and chain A')
    >>> coords = torch.tensor(coords[None, ...])
    >>> coords.shape
    torch.Size([1, 85, 3])
    >>> fragment = get_random_fragment(coords, fragment_ratio=0.5, noise=1.)
    >>> fragment.shape
    torch.Size([1, 42, 3])
    >>> fragment = get_random_fragment(coords, fragment_ratio=1.5, noise=1.)
    >>> fragment.shape
    torch.Size([1, 127, 3])

    >>> dmat = get_dmat(coords)
    >>> cmap = get_cmap(dmat)
    >>> dmat_fragment = get_dmat(fragment)
    >>> cmap_fragment = get_cmap(dmat_fragment)
    >>> coords = torch.randn(1, 100, 3)
    >>> _ = plt.subplot(1, 2, 1)
    >>> _ = plt.matshow(cmap.squeeze(), fignum=0)
    >>> _ = plt.colorbar()
    >>> _ = plt.subplot(1, 2, 2)
    >>> _ = plt.matshow(cmap_fragment.squeeze(), fignum=0)
    >>> _ = plt.colorbar()
    >>> plt.show()
    """
    if noise is None:
        noise = np.random.uniform()
    if fragment_ratio is None:
        fragment_ratio = np.random.uniform(low=0.25, high=2.)
    n = coords.shape[1]
    if fragment_ratio <= 1.:
        fragment_size = int(fragment_ratio * n)
    else:
        fragment_size = int((fragment_ratio - 1.) * n)
    if fragment_size <= 0:
        return coords + noise * torch.randn(coords.shape)
    i_start_max = n - fragment_size
    i_start = np.random.choice(i_start_max)
    i_stop = i_start + fragment_size
    inds = np.r_[i_start:i_stop]
    if fragment_ratio <= 1.:
        fragment = coords[:, inds, :]
    else:
        Rg = get_Rg(coords)
        v = torch.randn(1, 1, 3) * Rg
        to_add = coords[:, inds, :] + v
        fragment = torch.cat((coords, to_add), 1)
    fragment = fragment + noise * torch.randn(fragment.shape)
    return fragment


def normalize(inp):
    """
    >>> batch = 3
    >>> inp = 3. + 4 * torch.randn(batch, 1, 50, 50)
    >>> (torch.abs(inp.mean(dim=(2, 3)) - 3.) <= 5e-1).all()
    tensor(True)
    >>> (torch.abs(inp.std(dim=(2, 3)) - 4.) <= 5e-1).all()
    tensor(True)
    >>> out = normalize(inp)
    >>> out.shape
    torch.Size([3, 1, 50, 50])
    >>> (torch.abs(out.mean(dim=(2, 3))) < 1e-6).all()
    tensor(True)
    >>> (out.std(dim=(2, 3)) - 1.).mean() < 1e-6
    tensor(True)
    """
    mu = torch.mean(inp, dim=(2, 3))
    sigma = torch.std(inp, dim=(2, 3))
    if (sigma > 0).all():
        out = (inp - mu[..., None, None]) / sigma[..., None, None]
    else:
        out = (inp - mu[..., None, None])
    return out


def compute_pad(inp_size, out_size):
    """
    >>> A = torch.randn(78, 83)
    >>> out_size = (224, 224)
    >>> compute_pad(A.shape, out_size)
    (70, 71, 73, 73)
    """
    na, nb = inp_size
    nat, nbt = out_size
    assert na <= nat
    assert nb <= nbt
    narest, nbrest = nat - na, nbt - nb
    padtop = narest // 2
    padbottom = narest - padtop
    padleft = nbrest // 2
    padright = nbrest - padleft
    return (padleft, padright, padtop, padbottom)


def pad(mat, size):
    """
    >>> A = torch.randn(3, 1, 78, 83)
    >>> B = pad(A, size=(224, 224))
    >>> B.shape
    torch.Size([3, 1, 224, 224])
    """
    padlen = compute_pad(mat.shape[-2:], size)
    return torch.nn.functional.pad(mat, padlen)


def unpad(mat, size):
    """
    >>> A = torch.randn(3, 1, 78, 83)
    >>> B = pad(A, size=(224, 224))
    >>> B.shape
    torch.Size([3, 1, 224, 224])
    >>> A2 = unpad(B, (78, 83))
    >>> A2.shape
    torch.Size([3, 1, 78, 83])
    >>> (A2 == A).all()
    tensor(True)
    """
    na, nb = mat.shape[-2:]
    padlen = compute_pad(size, (na, nb))
    padleft, padright, padtop, padbottom = padlen
    return mat[..., padtop:na - padbottom, padleft:nb - padright]


def resize(mat, size):
    """
    >>> A = torch.randn(3, 1, 78, 78)
    >>> size = (224, 224)
    >>> B = resize(A, size=size)
    >>> B.shape
    torch.Size([3, 1, 224, 224])

    >>> A = torch.randn(3, 1, 278, 278)
    >>> B = resize(A, size=size)
    >>> B.shape
    torch.Size([3, 1, 224, 224])
    """
    na, nb = mat.shape[-2:]
    nat, nbt = size
    if na < nat and nb < nbt:
        out = pad(mat, size)
    elif na > nat and nb > nbt:
        out = torch.nn.functional.interpolate(mat, size=size)
    else:
        out = mat
    return out


def back_transform(mat, size):
    """
    >>> A = torch.randn(3, 1, 78, 78)
    >>> A_resize = resize(A, (224, 224))  # Here padding is applied
    >>> A_resize.shape
    torch.Size([3, 1, 224, 224])
    >>> A_back = back_transform(A_resize, (78, 78))  # and unpadding
    >>> A_back.shape
    torch.Size([3, 1, 78, 78])
    >>> (A_back == A).all()  # Therefore we retrieve the same A matrix
    tensor(True)

    >>> A = torch.randn(3, 1, 378, 378)
    >>> A_resize = resize(A, (224, 224))  # Here interpolation is applied to reduce the size
    >>> A_resize.shape
    torch.Size([3, 1, 224, 224])
    >>> A_back = back_transform(A_resize, (378, 378))  # and back interpolation
    >>> A_back.shape
    torch.Size([3, 1, 378, 378])
    >>> (A_back == A).all()  # therefore some information is lost
    tensor(False)
    """
    na, nb = mat.shape[-2:]
    nat, nbt = size
    if nat < na and nbt < nb:
        out = unpad(mat, size)
    elif nat > na and nbt > nb:
        out = torch.nn.functional.interpolate(mat, size=size)
    else:
        out = mat
    return out


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    import matplotlib.pyplot as plt
    ### UNCOMMENT FOR LOGGING ####
    import os
    import logging
    logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    logging.info(f"################ Starting {__file__} ################")
    ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
