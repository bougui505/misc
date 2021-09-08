#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
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


import numpy as np
import scipy.sparse


def get_indices(action, pos):
    pos = np.asarray(pos)
    actions = np.asarray(np.unravel_index(range(27), (3, 3, 3))).T - 1
    actions = actions[(actions != 0).any(axis=1)]
    return tuple(pos + actions[action])


class Gradient(object):
    def __init__(self, grid):
        self.grid = grid
        indices = np.asarray(np.unravel_index(range(grid.size), grid.shape)).T
        mask = (indices == 0).any(axis=1)
        self.shape = grid.shape
        n, p, q = self.shape
        mask = np.logical_or(mask, indices[:, 0] == n - 1)
        mask = np.logical_or(mask, indices[:, 1] == p - 1)
        mask = np.logical_or(mask, indices[:, 2] == q - 1)
        self.indices = indices[~mask]

    def _init_adj(self):
        """
        Initialize an empty adjacency matrix
        """
        n, p, q = self.shape
        adj = scipy.sparse.coo_matrix((np.prod(self.shape),) * 2)
        adj.data = np.zeros(n * p * q * 26)
        adj.row = np.zeros(n * p * q * 26, dtype=int)
        adj.col = np.zeros(n * p * q * 26, dtype=int)
        return adj

    def grad(self, grid=None, return_adj=False):
        """
        Compute the gradient with the 26 neighbors in a 3D grid.
        - Input: 3D grid of shape (n, p, q)
            - If input is a 4D grid of shape (26, n, p, q) (as the output)
            return_adj is set to True and return an adjacency matrix from the given gradient
        - Returns: ndarray with shape (26, n, p, q)
            - If return_adj is True, returns the gradient as an adjacency matrix of
        shape (n*p*q, n*p*q)
        """
        grad_to_adj = False
        if grid is None:
            grid = self.grid
        else:
            if grid.ndim == 4:
                assert grid.shape[0] == 26
                return_adj = True
                grad_to_adj = True
        if return_adj:
            data = []
            rows = []
            cols = []
        else:
            grad = []
        n, p, q = self.shape
        for action in range(26):
            neighbors = np.asarray(get_indices(action, self.indices))
            mask = None
            if not grad_to_adj:
                diff = (grid[tuple(neighbors.T)] - grid[tuple(self.indices.T)]).reshape((n - 2, p - 2, q - 2))
            else:
                inds = np.c_[[action, ] * len(self.indices), self.indices]
                vals = grid[tuple(inds.T)]
                mask = np.logical_or(np.isinf(vals), np.isnan(vals))
                inds = inds[~mask]
                vals = vals[~mask]
            if return_adj:
                i_inds = np.ravel_multi_index(tuple(neighbors.T), self.shape)
                j_inds = np.ravel_multi_index(tuple(self.indices.T), self.shape)
                if mask is not None:
                    i_inds = i_inds[~mask]
                    j_inds = j_inds[~mask]
                if not grad_to_adj:
                    data.extend(list(diff.flatten()))
                else:
                    data.extend(list(vals))
                rows.extend(list(i_inds))
                cols.extend(list(j_inds))
            else:
                grad_ = np.zeros_like(grid)
                grad_[1:-1, 1:-1, 1:-1] = diff
                grad.append(grad_)
        if return_adj:
            adj = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n * p * q, n * p * q))
            adj = adj.tocsr()
            return adj
        else:
            grad = np.asarray(grad)
            grad[np.isinf(grad)] = np.inf
            grad[np.isnan(grad)] = np.inf
            return grad


def gradtoadj(grad):
    """
    Convert a 3D-gradient to an adjacency matrix
    grad: gradient with shape (26, n, p, q)
    """
    gradfactory = Gradient(grad[0])
    adj = gradfactory.grad(grad)
    return adj


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()

    A = np.random.uniform(size=(10, 11, 12))
    gradfactory = Gradient(A)
    grad = gradfactory.grad()
    print(grad.shape)
    adj = gradfactory.grad(return_adj=True)
    print(adj.shape)
    adj2 = gradfactory.grad(grad)
    print(adj2.shape)
    print((adj2.todense() == adj.todense()).all())
    adj3 = gradtoadj(grad)
    print((adj3.todense() == adj.todense()).all())
