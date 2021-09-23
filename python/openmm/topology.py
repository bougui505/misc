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

from openmm.app.topology import Topology
from openmm.app.element import Element
from openmm.app.modeller import Modeller
import numpy as np


class System(Topology):
    def __init__(self):
        super().__init__()
        self.chain = None
        self.modeller = Modeller(super(), [])

    def add_chain(self, chainid):
        self.chain = super().addChain(id=chainid)

    def add_GLY(self, coords=np.zeros((4, 3))):
        residue = super().addResidue('GLY', self.chain)
        super().addAtom('N', Element.getByAtomicNumber(7), residue)
        super().addAtom('CA', Element.getByAtomicNumber(6), residue)
        super().addAtom('C', Element.getByAtomicNumber(6), residue)
        super().addAtom('O', Element.getByAtomicNumber(8), residue)
        super().createStandardBonds()
        if len(self.modeller.positions) > 0:
            self.modeller.positions = np.concatenate(
                (self.modeller.positions, coords))
        else:
            self.modeller.positions = coords
        assert super().getNumAtoms() == len(self.modeller.getPositions())


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()

    system = System()
    system.add_chain('A')
    system.add_GLY()
    print(system.getNumAtoms())
    print(system.getNumBonds())
    system.add_GLY()
    print(system.getNumAtoms())
    print(system.getNumBonds())
    # topology = Topology()
    # chain = topology.addChain(id="A")
    # build_GLY(topology, chain)
    # print(topology.getNumAtoms())
    # print(topology.getNumBonds())
