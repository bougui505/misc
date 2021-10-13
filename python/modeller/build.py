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

import modeller
import numpy as np
from modeller.optimizers import ConjugateGradients

lib = '/usr/lib/modeller10.1/modlib'

env = modeller.Environ()
env.libs.topology.read(file=f'{lib}/top_heav.lib')
env.libs.parameters.read(file=f'{lib}/par.lib')

modeller.log.none()


def extended(sequence):
    """

    Args:
        sequence: 1-letter code sequence (str)

    Returns:
        modeller Model object

    """

    mdl = modeller.Model(env)
    mdl.build_sequence(sequence)
    return mdl


class System():
    """
    Build a protein system by adding residues sequentially
    """
    def __init__(self):
        """
        >>> # Build a first system
        >>> system = System()
        >>> # with 2 glycine
        >>> system.add_residue('G')
        >>> system.add_residue('G')
        >>> # The following command create coordinates from internal coordinate system
        >>> system.build()
        Model containing 1 chain, 2 residues, and 9 atoms
        >>> # and evaluate its energy
        >>> system.energy[0]
        1.2566540241241455
        >>> system.coords
        array([[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00],
               [ 1.45529997e+00,  0.00000000e+00,  0.00000000e+00],
               [ 1.94122541e+00,  1.41604555e+00,  0.00000000e+00],
               [ 1.14785838e+00,  2.35453606e+00,  1.84466671e-07],
               [ 3.27375841e+00,  1.61899972e+00, -1.04427713e-07],
               [ 3.89874721e+00,  2.93403769e+00, -1.33662752e-08],
               [ 5.38628244e+00,  2.76508021e+00, -2.99142101e-07],
               [ 5.84848309e+00,  1.59291542e+00, -7.53011818e-07],
               [ 6.09959221e+00,  3.80372810e+00, -1.68363044e-07]])
        >>> # system.mdl.write('test.pdb')
        >>> # Try an other system
        >>> system = System()
        >>> # Still with 2 glycine, but specifying the position of the CA
        >>> system.add_residue('G', ca_coords=(0., 0., 0.))
        >>> system.add_residue('G', ca_coords=(3.8, 0., 0.))
        >>> system.build()
        Model containing 1 chain, 2 residues, and 9 atoms
        >>> # The energy is much higher as not optimal geometry due the the
        >>> # given CA position:
        >>> system.energy[0]
        353.2043762207031
        >>> # A quick energy minimization can quickly fix the geometry
        >>> system.minimize()
        >>> system.energy[0]
        11.27558422088623
        >>> # system.mdl.write('test2.pdb')
        """
        self.mdl = modeller.Model(env)
        self.sequence = ''
        self.initialize_xyz = True
        self.ca_coords = []

    def add_residue(self,
                    resname,
                    ca_coords=None,
                    patch_default=True,
                    blank_single_chain=False):
        """
        Add the given residue (given by resname) to the system

        Args:
            resname: Residue name to add in 1-letter code (str)

        """
        self.sequence += resname
        aln = modeller.Alignment(env)
        aln.append_sequence(self.sequence, blank_single_chain)
        self.mdl.clear_topology()
        self.mdl.generate_topology(aln[0],
                                   patch_default=patch_default,
                                   blank_single_chain=blank_single_chain)
        self.ca_coords.append(ca_coords)
        if ca_coords is not None:
            self.set_ca_coords()
            self.initialize_xyz = False

    def set_ca_coords(self):
        """
        Unset the coordinates of the non CA atoms
        """
        resid = 0
        for atom in self.mdl.atoms:
            if atom.name != 'CA':
                atom.x, atom.y, atom.z = -999.0, -999.0, -999.0
            else:
                if self.ca_coords[resid] is not None:
                    atom.x, atom.y, atom.z = self.ca_coords[resid]
                resid += 1

    def build(self):
        """
        Build the atomic coordinates of the system

        Returns:
            mdl: Modeller Model object

        """
        self.mdl.build(build_method='INTERNAL_COORDINATES',
                       initialize_xyz=self.initialize_xyz)
        atmsel = modeller.Selection(self.mdl)
        # This will setup the stereochemical energy (bonds, angles, dihedrals, impropers)
        # see: https://salilab.org/modeller/9.21/manual/node256.html
        self.mdl.restraints.make(atmsel,
                                 restraint_type='stereo',
                                 spline_on_site=False)
        self.mdl.restraints.make(atmsel,
                                 restraint_type='SPHERE14',
                                 spline_on_site=False)
        self.mdl.restraints.make(atmsel,
                                 restraint_type='LJ14',
                                 spline_on_site=False)
        # self.mdl.restraints.make(atmsel,
        #                          restraint_type='COULOMB14',
        #                          spline_on_site=False)
        self.mdl.restraints.make(atmsel,
                                 restraint_type='SPHERE',
                                 spline_on_site=False)
        self.mdl.restraints.make(atmsel,
                                 restraint_type='LJ',
                                 spline_on_site=False)
        # self.mdl.restraints.make(atmsel,
        #                          restraint_type='COULOMB',
        #                          spline_on_site=False)
        return self.mdl

    @property
    def energy(self):
        atmsel = modeller.Selection(self.mdl)
        energy, terms = atmsel.energy(output='NO_REPORT')
        return energy, terms

    def atoms(self):
        return self.mdl.atoms

    @property
    def coords(self):
        """
        Atomic coordinates of the system
        """
        out = []
        for atom in self.atoms():
            out.append([atom.x, atom.y, atom.z])
        return np.asarray(out)

    @property
    def ca_coords_min(self):
        """
        Atomic coordinates of C-alpha of the system
        """
        out = []
        for atom in self.atoms():
            if atom.name == 'CA':
                out.append([atom.x, atom.y, atom.z])
        return np.asarray(out)

    def atoms_defined(self):
        """
        List atoms with defined coordinates

        """
        selector = modeller.Selection(self.mdl)
        return selector.only_defined()

    def minimize(self, selection=None, max_iterations=20):
        cg = ConjugateGradients()
        if selection is None:
            atmsel = modeller.Selection(self.mdl)
        else:
            atmsel = modeller.Selection(selection)
        cg.optimize(atmsel, max_iterations=max_iterations)


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(
        description='Build an extended peptide using the given sequence')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-s', '--seq', help='1-letter code sequence to build')
    parser.add_argument('-o', '--out', help='out pdb file name')
    parser.add_argument('--test',
                        help='Test System class',
                        action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()

    mdl = extended(args.seq)
    mdl.write(args.out)
