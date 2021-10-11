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
import modeller


class Internal(object):
    """
    Get internal spherical coordinates of a protein C-alpha trace

    Attributes:
        coords: Input cartesian coords
        spherical: spherical internal coordinates
        resids: Index of internal coordinates

    """
    def __init__(self,
                 coords=None,
                 spherical=None,
                 modeller=False,
                 density=None):
        if density is not None:
            modeller = True
        if modeller:
            import misc.modeller.build as build
            self.system = build.System()
            self.env = build.env
            if density is not None:
                self.load_density(density)
        else:
            self.system = None
        self._coords = coords
        self._spherical = spherical
        self.resids = []
        if self._coords is not None:
            self._set()
        elif self._spherical is not None:
            self._back()

    def load_density(self, mrcfile):
        self.env.schedule_scale = modeller.physical.Values(em_density=10000)
        den = modeller.density(self.env,
                               file=mrcfile,
                               resolution=1.5,
                               em_density_format='CCP4')
        self.env.edat.density = den
        self.env.edat.dynamic_sphere = True

    def init_coords(self, coords, add=True):
        """

        Args:
            coords: coordinates to initialize the internal coordinate system (3 CA)
            add: If True, add the coordinates to the list of coordinates and compute
                 the basis from those coords. If False, only build the basis without
                 adding the coordinates.

        """
        if add:
            self._coords = []
            self._coords.extend([list(e) for e in coords])
        self._spherical = []
        self.basis = Basis()
        coords = np.asarray(coords)
        self.basis.build(coords)
        self.basis.change(coords)
        rthetaphi = self.basis.spherical
        self._spherical.extend([list(e) for e in rthetaphi])
        self.resids.extend(range(3))
        if self.system is not None:
            for ca_coord in coords:
                ca_coord = np.float_(ca_coord)
                self.system.add_residue('A', ca_coords=ca_coord)
            self.system.build()
        return rthetaphi

    def add_coords(self, coords, add=True):
        """
        Add a point to the set of internal coordinates

        Args:
            coords:

        """
        if add:
            self._coords.append(list(coords))
        self.basis = Basis()
        resid = self.resids[-1]
        self.basis.build(self.coords[resid - 2:resid + 1])
        self.basis.change(coords)
        rthetaphi = self.basis.spherical
        self._spherical.extend(list(rthetaphi))
        self.resids.append(resid + 1)
        if self.system is not None:
            self.system.add_residue('A', ca_coords=self.coords[-1])
            self.system.build()

    def _set(self):
        """
        Compute spherical internal coordinates (self.spherical) from Cartesian coordinates (self.coords)

        """
        n = len(self.coords)
        self.init_coords(self.coords[:3], add=False)
        for i in range(3, n):
            self.add_coords(self.coords[i], add=False)

    @property
    def spherical(self):
        return np.asarray(self._spherical)

    @property
    def coords(self):
        return np.asarray(self._coords, dtype=np.float)

    def init_spherical(self, rthetaphi):
        """

        Args:
            coords: spherical coordinates to initialize the coordinate system (3 CA)

        """
        self.basis = Basis(u=[1, 0, 0], v=[0, 1, 0], w=[0, 0, 1])
        self.basis.set_spherical(rthetaphi)
        self._coords.extend([list(e) for e in self.basis.coords])
        self.resids.extend(range(3))

    def add_spherical(self, rthetaphi):
        self.basis = Basis()
        resid = self.resids[-1]
        self.basis.build(self.coords[resid - 2:resid + 1])
        self.basis.set_spherical(rthetaphi)
        self._coords.extend(list(self.basis.coords))
        self.resids.append(resid + 1)
        if self.system is not None:
            self.system.add_residue('A', ca_coords=self.coords[-1])
            self.system.build()

    def _back(self):
        """
        Compute Cartesian coordinates (self.coords) from spherical internal coordinates (self.spherical)

        """
        n = len(self.spherical)
        self._coords = []
        self.init_spherical(self.spherical[:3])
        for i in range(3, n):
            self.add_spherical(self.spherical[i])

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
    parser.add_argument('--plot',
                        help='Plot energy diagram for spherical coordinates',
                        action='store_true')
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
    if args.plot:
        import matplotlib.pyplot as plt
        init_ca = np.asarray([[8.504, 6.440, 7.674], [7.874, 10.070, 8.635],
                              [11.095, 11.990, 9.462]])
        n = 25
        emap = np.zeros((n, 2 * n))
        # Read an EM-density map to compute density energy term
        internal = Internal(modeller=True, density='data/1igd_center.mrc')
        min_energy = np.inf
        for i, theta in enumerate(np.linspace(0, np.pi, num=n)):
            for j, phi in enumerate(np.linspace(0, 2 * np.pi, num=2 * n)):
                internal = Internal(modeller=True)
                internal.init_coords(init_ca)
                internal.add_spherical([3.8, theta, phi])
                internal.system.minimize()
                em_density_energy = internal.system.energy[1][
                    modeller.physical.em_density]
                # emap[i, j] = em_density_energy
                energy = internal.system.energy[0]
                if energy < min_energy:
                    min_energy = energy
                    internal.system.mdl.write('min.pdb')
                emap[i, j] = energy
        print(min_energy)
        im = plt.matshow(emap, origin='lower', extent=[0, 360, 0, 180])
        plt.colorbar()
        plt.xlabel('φ (deg.)')
        plt.ylabel('θ (deg.)')
        plt.show()
