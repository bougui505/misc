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

# Based on the openmm-plumed Plugin (https://github.com/openmm/openmm-plumed)

import openmm.app as app
import openmm
import openmmplumed
import openmm.unit as unit
import os
from sys import stdout


def add_plumed_forces(plumed_script, system):
    with open(plumed_script, 'r') as plumedfile:
        script = plumedfile.read()
    system.addForce(openmmplumed.PlumedForce(script))


def run(inpdb='input.pdb',
        plumed_script=None,
        forcefield_xml='amber99sb.xml',
        watermodel_xml='tip3p.xml',
        temperature=300 * unit.kelvin,
        frictionCoeff=1 / unit.picosecond,
        stepSize=0.002 * unit.picoseconds,
        outbasename='output',
        reportInterval=1000,
        steps=10000,
        pH=7.0,
        padding=1.0 * unit.nanometers):
    pdb = app.PDBFile(inpdb)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    forcefield = app.ForceField(forcefield_xml, watermodel_xml)
    modeller.addHydrogens(forcefield, pH=pH)
    modeller.addSolvent(forcefield, padding=padding)
    system = forcefield.createSystem(modeller.topology,
                                     nonbondedMethod=app.PME,
                                     nonbondedCutoff=1 * unit.nanometer,
                                     constraints=app.HBonds)
    integrator = openmm.LangevinIntegrator(temperature, frictionCoeff,
                                           stepSize)
    # ######## add PLUMED forces ##########
    if plumed_script is not None:
        add_plumed_forces(plumed_script, system)
    # #####################################
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()
    with open(f'{outbasename}_start.pdb', 'w') as outpdbfile:
        app.PDBFile.writeFile(modeller.topology, modeller.positions,
                              outpdbfile)
    outdcd = f'{outbasename}_traj.dcd'
    simulation.reporters.append(app.DCDReporter(outdcd, reportInterval))
    simulation.reporters.append(
        app.CheckpointReporter(f'{outbasename}.chk', steps // 100))
    simulation.reporters.append(
        app.StateDataReporter(stdout,
                              reportInterval,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              temperature=True,
                              speed=True,
                              remainingTime=True,
                              totalSteps=steps))
    simulation.step(steps)


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb')
    parser.add_argument('--nsteps', type=int)
    parser.add_argument('--plumed',
                        help='Plumed script (see example in plumed.dat)')
    args = parser.parse_args()
    outbasename = os.path.splitext(args.pdb)[0]
    run(inpdb=args.pdb,
        plumed_script=args.plumed,
        steps=args.nsteps,
        outbasename=outbasename)
