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
from openmm.app.forcefield import ForceField
from openmm.app.element import Element
from openmm.app.modeller import Modeller
from openmm.app.simulation import Simulation
from openmm import LangevinIntegrator
import simtk.unit as unit
import numpy as np


class MetaSystem(Topology):
    def __init__(self, xmlff='amber14-all.xml'):
        super().__init__()
        self.chain = None
        self.modeller = Modeller(super(), [])
        self.forcefield = ForceField(xmlff)
        self.integrator = LangevinIntegrator(300 * unit.kelvin,
                                             1 / unit.picosecond,
                                             0.004 * unit.picoseconds)

    def add_chain(self, chainid):
        self.chain = super().addChain(id=chainid)

    def add_GLY(self, coords=np.random.uniform(low=0., high=1., size=(4, 3))):
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

    def add_OXT(self, coords=np.random.uniform(low=0., high=1., size=(1, 3))):
        residue = list(self.modeller.topology.residues())[-1]
        super().addAtom('OXT', Element.getByAtomicNumber(8), residue)
        super().createStandardBonds()
        self.modeller.positions = np.concatenate(
            (self.modeller.positions, coords))
        assert super().getNumAtoms() == len(self.modeller.getPositions())

    def createSystem(self):
        self.add_OXT()
        self.modeller.positions = unit.Quantity(self.modeller.positions,
                                                unit.nanometer)
        self.modeller.addHydrogens(self.forcefield)
        self.system = self.forcefield.createSystem(self.modeller.topology)
        return self.system

    def minimize(self):
        simulation = Simulation(super(), self.system, self.integrator)
        simulation.context.setPositions(self.modeller.positions)
        simulation.minimizeEnergy()


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    args = parser.parse_args()

    metasystem = MetaSystem()
    metasystem.add_chain('A')
    metasystem.add_GLY()
    print(metasystem.getNumAtoms())
    print(metasystem.getNumBonds())
    metasystem.add_GLY()
    print(metasystem.getNumAtoms())
    print(metasystem.getNumBonds())
    system = metasystem.createSystem()
    metasystem.minimize()
