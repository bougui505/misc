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


def get_dmat(coords):
    """
    """
    dmat = torch.cdist(coords, coords)
    dmat = dmat[:, None, ...]  # Add the channel dimension
    return dmat


class Normalizer(object):
    def __init__(self, batch):
        """
        >>> batch = [1 + torch.randn(1, 1, 249, 249), 2 + 2* torch.randn(1, 1, 639, 639), 3 + 3 * torch.randn(1, 1, 390, 390), 4 + 4 * torch.randn(1, 1, 131, 131)]
        >>> normalizer = Normalizer(batch)
        >>> [torch.round(e) for e in normalizer.mu]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        >>> [torch.round(e) for e in normalizer.sigma]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        >>> out = normalizer.transform(batch)
        >>> [torch.round(e.mean()).abs() for e in out]
        [tensor(0.), tensor(0.), tensor(0.), tensor(0.)]
        >>> [torch.round(e.std()) for e in out]
        [tensor(1.), tensor(1.), tensor(1.), tensor(1.)]
        >>> x = normalizer.inverse_transform(out)
        >>> [torch.round(e.mean()) for e in x]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        >>> [torch.round(e.std()) for e in x]
        [tensor(1.), tensor(2.), tensor(3.), tensor(4.)]
        """
        self.batch = [e for e in batch if e is not None]
        self.mu = torch.tensor([e.mean() for e in self.batch])
        self.sigma = torch.tensor([e.std() for e in self.batch])

    def transform(self, x):
        n = len(x)
        out = []
        for i in range(n):
            if self.sigma[i] > 0:
                out.append((x[i] - self.mu[i]) / self.sigma[i])
            else:
                out.append(x[i] - self.mu[i])
        return out

    def inverse_transform(self, x):
        n = len(x)
        out = []
        for i in range(n):
            out.append(x[i] * self.sigma[i] + self.mu[i])
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
