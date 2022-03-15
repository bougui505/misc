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


class Rotation(torch.nn.Module):
    """
    See: https://stackoverflow.com/a/61242932/1679629
    >>> A = torch.rand((10, 3))
    >>> A.shape
    torch.Size([10, 3])
    >>> rotation = Rotation()
    >>> B = rotation(A)
    >>> B.shape
    torch.Size([10, 3])
    """
    def __init__(self, requires_grad=True):
        super().__init__()
        self.roll = torch.nn.Parameter(torch.randn(1, requires_grad=requires_grad))
        self.yaw = torch.nn.Parameter(torch.randn(1, requires_grad=requires_grad))
        self.pitch = torch.nn.Parameter(torch.randn(1, requires_grad=requires_grad))
        self.tensor_0 = torch.zeros(1)
        self.tensor_1 = torch.ones(1)

    def __call__(self, x):
        RX = torch.stack([
            torch.stack([self.tensor_1, self.tensor_0, self.tensor_0]),
            torch.stack([self.tensor_0, torch.cos(self.roll), -torch.sin(self.roll)]),
            torch.stack([self.tensor_0, torch.sin(self.roll), torch.cos(self.roll)])
        ]).reshape(3, 3)
        RY = torch.stack([
            torch.stack([torch.cos(self.pitch), self.tensor_0,
                         torch.sin(self.pitch)]),
            torch.stack([self.tensor_0, self.tensor_1, self.tensor_0]),
            torch.stack([-torch.sin(self.pitch), self.tensor_0,
                         torch.cos(self.pitch)])
        ]).reshape(3, 3)
        RZ = torch.stack([
            torch.stack([torch.cos(self.yaw), -torch.sin(self.yaw), self.tensor_0]),
            torch.stack([torch.sin(self.yaw), torch.cos(self.yaw), self.tensor_0]),
            torch.stack([self.tensor_0, self.tensor_0, self.tensor_1])
        ]).reshape(3, 3)
        R = torch.mm(RZ, RY)
        R = torch.mm(R, RX)
        out = (R.mm(x.T)).T
        return out


def loss_MSE(A, B):
    return ((B - A)**2).mean()


class FlexFitter(torch.nn.Module):
    """
    Torch model (https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)
    >>> A = torch.rand((10, 3))
    >>> A.shape
    torch.Size([10, 3])
    >>> ff = FlexFitter()
    >>> x = ff(A)
    >>> x.shape
    torch.Size([10, 3])
    >>> print(len(list(ff.parameters())))
    3

    """
    def __init__(self, dorotate=True):
        super(FlexFitter, self).__init__()
        self.dorotate = dorotate
        self.rotation = Rotation()

    def forward(self, x):
        if self.dorotate:
            x = self.rotation(x)
        return x


def fit(inp, target, maxiter, stop=1e-3, verbose=True):
    """
    >>> inp = torch.rand((10, 3))
    >>> rotation = Rotation()
    >>> target = rotation(inp)
    >>> loss_init = loss_MSE(inp, target)
    >>> output, loss = fit(inp, target, maxiter=10000, verbose=False)
    >>> loss_init, loss
    """
    ff = FlexFitter()
    optimizer = torch.optim.Adam(ff.parameters())
    if verbose:
        pbar = tqdm.tqdm(total=maxiter)
    for i in range(maxiter):
        optimizer.zero_grad()
        output = ff(inp)
        loss = loss_MSE(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if verbose:
            pbar.update(1)
        if loss < stop:
            break
    return output, loss


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
