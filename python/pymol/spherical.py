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
from pymol import cmd
from misc.basis import Basis


class Internal(object):
    """
    Get internal spherical coordinates of a protein C-alpha trace

    Attributes:
        spherical_coords: spherical internal coordinates
        initcoords: first cartesian atomic coordinates to initialize the internal coordinate system
        coords: Input cartesian coords
        inds: Index of internal coordinates

    """
    def __init__(self, coords):
        self.coords = coords
        self._set()

    def _set(self):
        n = len(self.coords)
        r, theta, phi = [], [], []
        inds = []
        for i in range(n - 3):
            window = self.coords[i:i + 4]
            basis = Basis()
            basis.build(window[:3])
            if i == 0:
                basis.change(window[:3])
                init_coords = basis.spherical
                r.extend(init_coords[:, 0])
                theta.extend(init_coords[:, 1])
                phi.extend(init_coords[:, 2])
                inds.extend(range(3))
            basis.change(window[3])
            rthetaphi = basis.spherical
            r.extend(rthetaphi[:, 0])
            theta.extend(rthetaphi[:, 1])
            phi.extend(rthetaphi[:, 2])
            inds.append(i + 3)
        self.spherical_coords = np.c_[r, theta, phi]
        self.inds = inds

    def write(self, outputfilename):
        out = np.c_[self.inds, self.spherical_coords]
        np.savetxt('internal_ca_coords.txt',
                   out,
                   header='#ind #r #theta #phi',
                   fmt='%d %.3f %.3f %.3f',
                   comments='')


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='pdb protein structure file name')
    args = parser.parse_args()

    cmd.load(args.pdb, object='inp')
    coords = cmd.get_coords(selection='name CA')
    internal = Internal(coords)
    internal.write('internal_ca_coords.txt')
