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
        _coords: internal list to store coordinates
        coords: Input cartesian coords
        _spherical: internal list to store spherical internal coordinates
        spherical: spherical internal coordinates
        resids: Index of internal coordinates
        system: sytem (Modeller)
        env: Modeller environment
        basis: Basis object to store the internal coordinates basis of the step

    >>> internal = Internal(modeller=True)  # doctest:+ELLIPSIS
    <BLANKLINE>
                             MODELLER 10.1, 2021/03/12, r12156
    <BLANKLINE>
         PROTEIN STRUCTURE MODELLING BY SATISFACTION OF SPATIAL RESTRAINTS
    <BLANKLINE>
    <BLANKLINE>
                         Copyright(c) 1989-2021 Andrej Sali
                                All Rights Reserved
    <BLANKLINE>
                                 Written by A. Sali
                                   with help from
                  B. Webb, M.S. Madhusudhan, M-Y. Shen, G.Q. Dong,
              M.A. Marti-Renom, N. Eswar, F. Alber, M. Topf, B. Oliva,
                 A. Fiser, R. Sanchez, B. Yerkovich, A. Badretdinov,
                         F. Melo, J.P. Overington, E. Feyfant
                     University of California, San Francisco, USA
                        Rockefeller University, New York, USA
                          Harvard University, Cambridge, USA
                       Imperial Cancer Research Fund, London, UK
                  Birkbeck College, University of London, London, UK
    <BLANKLINE>
    <BLANKLINE>
    Kind, OS, HostName, Kernel, Processor: 4, Linux mantrisse 5.10.0-8-amd64 x86_64
    Date and time of compilation         : 2021/03/12 00:18:43
    MODELLER executable type             : x86_64-intel8
    Job starting time (YY/MM/DD HH:MM:SS): ...
    <BLANKLINE>
    read_to_681_> topology.submodel read from topology file:        3
    >>> internal.add_basis([(8.482, 5.881, 6.315), (8.504, 6.440, 7.674), (8.417, 7.966, 7.566)])
    >>> internal.system.energy[0]
    4.497385501861572
    >>> internal.add_basis([(7.914, 8.587, 8.600), (7.874, 10.070, 8.635), (9.218, 10.455, 9.275)])
    >>> init_ca = np.asarray([[8.504, 6.440, 7.674], [7.874, 10.070, 8.635], [11.095, 11.990, 9.462]])
    >>> internal.system.energy[0]
    13.126262664794922
    >>> internal.system.minimize()
    >>> internal.system.energy[0]
    7.411508083343506
    >>> # Working with Modeller and EM-density
    >>> internal = Internal(modeller=True, density='data/1igd_center.mrc', density_weight=100.)  # doctest:+ELLIPSIS
    >>> # internal = Internal(modeller=True)
    >>> rthetaphi = internal.init_coords(init_ca)
    >>> internal.add_spherical([3.8, np.pi/2, np.pi/6])
    >>> internal.system.minimize()
    >>> # Check if the EM density is effectively loaded:
    >>> em_density_energy = internal.system.energy[1][modeller.physical.em_density]
    >>> print(em_density_energy)
    -18.308961868286133
    >>> total_energy = internal.system.energy[0]
    >>> print(total_energy)
    27.904043197631836
    >>> internal.system.minimize()
    >>> total_energy = internal.system.energy[0]
    >>> print(total_energy)
    15.505636215209961
    >>> internal.coords
    array([[ 8.504     ,  6.44      ,  7.674     ],
           [ 7.874     , 10.07      ,  8.635     ],
           [11.095     , 11.99      ,  9.462     ],
           [12.83405143, 15.19958501, 10.51758712]])

    """
    def __init__(self,
                 coords=None,
                 spherical=None,
                 modeller=False,
                 density=None,
                 density_weight=1.):
        if density is not None:
            modeller = True
        if modeller:
            import misc.modeller.build as build
            self.env = build.env
            if density is not None:
                self.load_density(mrcfile=density,
                                  density_weight=density_weight)
            self.system = build.System()
        else:
            self.system = None
        self._coords = coords
        self._spherical = spherical
        self.resids = []
        if self._coords is not None:
            self._set()
        elif self._spherical is not None:
            self._back()

    def load_density(self, mrcfile, density_weight=1.):
        self.env.schedule_scale = modeller.physical.Values(
            em_density=density_weight)
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

    def add_basis(self, coords, add=True):
        """
        Add 3 points defining a new internal basis

        Args:
            coords: coordinates of 3 points in space (shape: 3,3)
        """
        coords = np.asarray(coords)
        assert coords.shape == (
            3, 3
        ), f"3 points required to construct an internal basis (input shape: {coords.shape})"
        if add:
            if self._coords is None:
                self._coords = []
            self._coords.extend([list(a) for a in coords])
        self.basis = Basis()
        if len(self.resids) > 0:
            resid = self.resids[-1]
        else:
            resid = 0
        self.basis.build(coords, origin='center')
        self.basis.change(coords)
        rthetaphi = self.basis.spherical
        if self._spherical is None:
            self._spherical = []
        self._spherical.extend(list(rthetaphi))
        self.resids.extend([
            resid + 1,
        ] * 3)
        if self.system is not None:
            self.system.add_residue('A', bb_coords=coords)
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
    import sys
    import doctest
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
    parser.add_argument('--test',
                        help='Test System class',
                        action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()

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
        internal = Internal(modeller=True,
                            density='data/1igd_center.mrc',
                            density_weight=10000.)
        min_energy = np.inf
        for i, theta in enumerate(np.linspace(0, np.pi, num=n)):
            for j, phi in enumerate(np.linspace(0, 2 * np.pi, num=2 * n)):
                internal = Internal(modeller=True)
                internal.init_coords(init_ca)
                internal.add_spherical([3.8, theta, phi])
                internal.system.minimize()
                em_density_energy = internal.system.energy[1][
                    modeller.physical.em_density]
                print(em_density_energy)
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
