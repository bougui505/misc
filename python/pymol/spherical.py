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
        coords: Input cartesian coords
        spherical: spherical internal coordinates
        resids: Index of internal coordinates

    """
    def __init__(self, coords=None, spherical=None):
        self.coords = coords
        self._spherical = spherical
        self.resids = []
        if self.coords is not None:
            self._set()
        elif self._spherical is not None:
            self._back()

    def init_coords(self, coords):
        """

        Args:
            coords: coordinates to initialize the internal coordinate system (3 CA)

        """
        basis = Basis()
        basis.build(coords)
        basis.change(coords)
        rthetaphi = basis.spherical
        self._spherical.extend([list(e) for e in rthetaphi])
        self.resids.extend(range(3))
        return rthetaphi

    def add_coords(self, coords):
        basis = Basis()
        resid = self.resids[-1]
        basis.build(self.coords[resid - 2:resid + 1])
        basis.change(coords)
        rthetaphi = basis.spherical
        self._spherical.extend(list(rthetaphi))
        self.resids.append(resid + 1)

    def _set(self):
        """
        Compute spherical internal coordinates (self.spherical) from Cartesian coordinates (self.coords)

        """
        n = len(self.coords)
        self._spherical = []
        self.init_coords(self.coords[:3])
        for i in range(3, n):
            self.add_coords(self.coords[i])

    @property
    def spherical(self):
        return np.asarray(self._spherical)

    def _back(self):
        """
        Compute Cartesian coordinates (self.coords) from spherical internal coordinates (self.spherical)

        """
        n = len(self.spherical)
        x, y, z = [], [], []
        for i in range(n - 3):
            window = self.spherical[i:i + 4]
            if i == 0:
                basis = Basis(u=[1, 0, 0], v=[0, 1, 0], w=[0, 0, 1])
                basis.set_spherical(window[:3])
                x.extend(basis.coords[:, 0])
                y.extend(basis.coords[:, 1])
                z.extend(basis.coords[:, 2])
            basis = Basis()
            coords = np.c_[x[-3:], y[-3:], z[-3:]]
            basis.build(coords[:3])
            basis.set_spherical(window[3])
            x.extend(basis.coords[:, 0])
            y.extend(basis.coords[:, 1])
            z.extend(basis.coords[:, 2])
            self.resids.append(i + 3)
        self.coords = np.c_[x, y, z]

    def write(self, outputfilename):
        out = np.c_[self.resids, self.spherical]
        np.savetxt('internal_ca_coords.txt',
                   out,
                   header='#ind #r #theta #phi',
                   fmt='%d %.3f %.3f %.3f',
                   comments='')

    def write_pdb(self, outputfilename):
        cmd.reinitialize()
        for i, coord in enumerate(self.coords):
            resi = i + 1
            cmd.pseudoatom('catrace',
                           name='CA',
                           resi=resi,
                           elem='C',
                           pos=tuple(coord),
                           hetatm=0)
            if i > 0:
                sel1 = f'resi {resi - 1} and catrace'
                sel2 = f'resi {resi} and catrace'
                cmd.bond(sel1, sel2)
        cmd.set('pdb_conect_all', 1)
        cmd.save(outputfilename, selection='catrace')


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='pdb protein structure file name')
    parser.add_argument('--internal',
                        help='Internal coordinates to convert to Cartesian')
    args = parser.parse_args()

    if args.pdb is not None:
        cmd.load(args.pdb, object='inp')
        coords = cmd.get_coords(selection='name CA')
        internal = Internal(coords=coords)
        internal.write('internal_ca_coords.txt')
    if args.internal is not None:
        spherical_coords = np.genfromtxt(args.internal, usecols=(1, 2, 3))
        internal = Internal(spherical=spherical_coords)
        internal.write_pdb('trace.pdb')
