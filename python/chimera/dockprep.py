#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Jun  5 09:46:07 2024
# Usage: chimera --nogui file.pdb dockprep.py outfilename.mol2

import sys

import chimera
from chimera import runCommand
from DockPrep import prep

outfile = sys.argv[-1]

models = chimera.openModels.list(modelTypes=[chimera.Molecule])
runCommand('delete @H')
prep(models)
from WriteMol2 import writeMol2

writeMol2(models, outfile)
print("Output written in:", outfile)
