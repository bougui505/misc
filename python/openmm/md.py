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
try:
    import openmmplumed
except ImportError:
    print('Cannot import openmmplumed')
import openmm.unit as unit
import os
from sys import stdout
import time
import shutil
import sys


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
        padding=1.0 * unit.nanometers,
        boxSize=None,
        checkpoint=None,
        implicit_solvent=False):
    """
    inpdb: input pdb file
    plumed_script: if not None, run a metadynamics using the given plumed script in PLUMED using openmmplumed
    checkpoint: if not None, resume the simulation by loading the given checkpoint
    implicit_solvent: if True, use the amber99_obc implicit solvation model
    """
    if checkpoint is not None:
        restart = True
    else:
        restart = False
    outdcd = f'{outbasename}_traj.dcd'
    outcheckpoint = f'{outbasename}.chk'
    outlog = f'{outbasename}.log'
    if restart:
        inpdb = f'{outbasename}_start.pdb'
    pdb = app.PDBFile(inpdb)
    print('Creating modeller object...')
    modeller = app.Modeller(pdb.topology, pdb.positions)
    if implicit_solvent:
        watermodel_xml = 'implicit/obc1.xml'
    forcefield = app.ForceField(forcefield_xml, watermodel_xml)
    if not restart:
        print('Adding hydrogens...')
        modeller.addHydrogens(forcefield, pH=pH)
        if not implicit_solvent:
            print('Adding solvent...')
            if boxSize is not None:
                modeller.addSolvent(forcefield, boxSize=boxSize)
            else:
                modeller.addSolvent(forcefield, padding=padding)
    print('Creating system...')
    if implicit_solvent:
        system = forcefield.createSystem(modeller.topology,
                                         soluteDielectric=1.0,
                                         solventDielectric=80.0,
                                         constraints=app.HBonds)
    else:
        system = forcefield.createSystem(modeller.topology,
                                         nonbondedMethod=app.PME,
                                         nonbondedCutoff=1 * unit.nanometer,
                                         constraints=app.HBonds)
    integrator = openmm.LangevinIntegrator(temperature, frictionCoeff, stepSize)
    # ######## add PLUMED forces ##########
    if plumed_script is not None:
        print('Adding plumed forces...')
        add_plumed_forces(plumed_script, system)
    # #####################################
    simulation = app.Simulation(modeller.topology, system, integrator)
    if not restart:
        simulation.context.setPositions(modeller.positions)
        print('Minimizing energy...')
        simulation.minimizeEnergy()
        with open(f'{outbasename}_start.pdb', 'w') as outpdbfile:
            app.PDBFile.writeFile(modeller.topology, modeller.positions, outpdbfile)
    else:  # Restart a simulation
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_dir = f'md_{timestamp}'
        os.mkdir(backup_dir)
        for filename in [outdcd, outlog]:
            try:
                shutil.move(filename, backup_dir)
            except FileNotFoundError:
                print(f'{filename} is not present')
        shutil.copy(outcheckpoint, backup_dir)
        if plumed_script is not None:
            shutil.copy('HILLS', backup_dir)
        simulation.loadCheckpoint(checkpoint)
    simulation.reporters.append(app.DCDReporter(outdcd, reportInterval))
    simulation.reporters.append(app.CheckpointReporter(outcheckpoint, steps // 100))
    simulation.reporters.append(
        app.StateDataReporter(outlog,
                              reportInterval,
                              step=True,
                              time=True,
                              potentialEnergy=True,
                              temperature=True,
                              speed=True,
                              remainingTime=True,
                              totalSteps=steps))
    print('Running MD...')
    simulation.step(steps)


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb')
    parser.add_argument('--nsteps', type=int)
    parser.add_argument('-ff',
                        '--forcefield',
                        help='xml file for the forcefield to use. (default: amber99sb.xml)',
                        default='amber99sb.xml')
    parser.add_argument('--watermodel',
                        help='xml file for the watermodel to use. (default: tip3p.xml)',
                        default='tip3p.xml')
    parser.add_argument('--temperature', help='Temperature in Kelvin (default 300 K)', default=300., type=float)
    parser.add_argument('--padding', help='Padding in nanometers for the box (default: 1 nm)', default=1.0, type=float)
    parser.add_argument('--box',
                        help='Size of the box in nm (optional. If not given see --padding)',
                        metavar=('NX', 'NY', 'NZ'),
                        default=None,
                        type=float,
                        nargs=3)
    parser.add_argument('--plumed', help='Plumed script (see example in plumed.dat)')
    parser.add_argument('--restart', help='Restart the simulation. Give the checkpoint file as argument')
    parser.add_argument('--report_interval',
                        help='Report interval for the log and the trajectory (default 1000)',
                        default=1000,
                        type=int)
    parser.add_argument('--implicit_solvent', action='store_true', help="Use 'amber99_obc' implicit solvation model")
    args = parser.parse_args()
    outbasename = os.path.splitext(args.pdb)[0]
    padding = args.padding * unit.nanometers
    temperature = args.temperature * unit.kelvin
    if args.box is None:
        boxSize = None
    else:
        nx, ny, nz = args.box
        boxSize = openmm.Vec3(nx, ny, nz) * unit.nanometers
    run(inpdb=args.pdb,
        temperature=temperature,
        forcefield_xml=args.forcefield,
        watermodel_xml=args.watermodel,
        padding=padding,
        boxSize=boxSize,
        plumed_script=args.plumed,
        steps=args.nsteps,
        outbasename=outbasename,
        checkpoint=args.restart,
        reportInterval=args.report_interval,
        implicit_solvent=args.implicit_solvent)
