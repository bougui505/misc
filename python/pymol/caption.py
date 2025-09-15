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
# AI! when using cmd.fetch, the pdb must be downloaded in PDB_DOWNLOAD_PATH

def loader(pdb):
    pass
