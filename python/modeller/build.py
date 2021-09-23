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
        >>> system = System()
        >>> system.add_residue('A')
        >>> system.add_residue('D')
        >>> system.build()
        Model containing 1 chain, 2 residues, and 14 atoms
        >>> system.energy
        40331.953125
        """
        self.mdl = modeller.Model(env)
        self.sequence = ''

    def add_residue(self, resname):
        """
        Add the given residue (given by resname) to the system

        Args:
            resname: Residue name to add in 1-letter code (str)

        """
        self.sequence += resname
        aln = modeller.Alignment(env)
        aln.append_sequence(self.sequence)
        self.mdl.clear_topology()
        self.mdl.generate_topology(aln[0])

    def build(self):
        """
        Build the atomic coordinates of the system

        Returns:
            mdl: Modeller Model object

        """
        self.mdl.build(build_method='INTERNAL_COORDINATES',
                       initialize_xyz=False)
        return self.mdl

    @property
    def energy(self):
        atmsel = modeller.Selection(self.mdl)
        energy, terms = atmsel.energy(output='NO_REPORT', file='energy.log')
        return energy

    def minimize(self):
        cg = ConjugateGradients()
        atmsel = modeller.Selection(self.mdl)
        cg.optimize(atmsel, max_iterations=20)


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
