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


class Box(object):
    """
    Encode a box for coordinates
    """
    def __init__(self, shape, padding=0, padded_shape=False):
        """
        >>> box = Box((10, 12))
        >>> box.bounding_coords((3, 8))
        array([3, 8])
        >>> box.bounding_coords((-3, 15))
        array([ 0, 11])
        >>> box.bounding_coords((10, 12))
        array([ 9, 11])
        >>> box = Box((10, 12), padding=5)
        >>> box.shape
        array([20, 22])
        >>> box.bounding_coords((8, 10))  # Unpadded coords
        array([13, 15])
        >>> box.bounding_coords((-3, 15))
        array([ 5, 16])
        >>> box.bounding_coords((2, 15), padded=True)
        array([ 5, 15])
        >>> box.bounding_coords((10, 12), padded=False)
        array([14, 16])
        >>> box = Box((20, 22), padding=5, padded_shape=True)
        >>> box.bounding_coords((8, 15), padded=False)
        array([13, 16])
        """
        self.shape = np.asarray(shape)
        self.dim = len(shape)
        if isinstance(padding, int):
            self.padding = np.asarray([
                padding,
            ] * self.dim)
        else:
            self.padding = padding
        if not padded_shape:
            self.shape += 2 * self.padding

    def bounding_coords(self, coords, padded=False):
        """
        if padded is False:
            coords are given in the frame of the box without padding
        else:
            coords are given in the padded frame
        Returns:
            bounded coords in the padded frame
        """
        coords = np.asarray(coords)
        if not padded:
            coords += self.padding
        assert coords.ndim == self.shape.ndim
        lower_bound = (coords < self.padding)
        self.bounded = False
        if lower_bound.any():
            coords[lower_bound] = self.padding[lower_bound]
            self.bounded = True
        upper_bound = (coords >= self.shape - self.padding)
        if upper_bound.any():
            coords[upper_bound] = (self.shape - self.padding)[upper_bound] - 1
            self.bounded = True
        return coords


if __name__ == '__main__':
    import argparse
    import doctest
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='test the module', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
