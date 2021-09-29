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


class Basis():
    def __init__(self, u, v, w):
        """

        Args:
            u: x axis of new basis in the old basis
            v: y axis of new basis in the old basis
            w: z axis of new basis in the old basis

        >>> u = [1, 0, 0]
        >>> v = [0, 1, 0]
        >>> w = [0, 0, 1]
        >>> basis = Basis(v, w, u)
        >>> coords = np.arange(15).reshape((5, 3))
        >>> coords
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11],
               [12, 13, 14]])
        >>> basis.change(coords)
        array([[ 1.,  2.,  0.],
               [ 4.,  5.,  3.],
               [ 7.,  8.,  6.],
               [10., 11.,  9.],
               [13., 14., 12.]])

        """
        self.dim = len(u)
        self.u, self.v, self.w = u, v, w
        self.A = np.c_[self.u, self.v, self.w]  # transition matrix
        assert (self.A.T.dot(self.A) == np.identity(
            self.dim)).all(), "u, v, w are not an orthonormal basis"
        self.A_inv = np.linalg.inv(self.A)

    def change(self, coords):
        """

        Args:
            coords: Coordinates of points in the old basis (shape: (n, self.dim))

        """
        coords_new = self.A_inv.dot(coords.T).T
        return coords_new


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the module', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
