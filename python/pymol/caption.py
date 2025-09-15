#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Sep 15 09:58:44 2025

from pymol import cmd
import os

PDB_DOWNLOAD_PATH = os.path.expanduser("~/pdb")

def loader(pdb):
    if not os.path.isfile(pdb):
        cmd.fetch(pdb, path=PDB_DOWNLOAD_PATH)
    else:
        cmd.load(pdb)

def sphere(selection='all', padding=5.0, npts=100):
    # AI! define npts points equally spaced on a sphere that encompass the protein + padding
