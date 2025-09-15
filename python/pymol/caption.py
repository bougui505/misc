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
import numpy as np

PDB_DOWNLOAD_PATH = os.path.expanduser("~/pdb")

def loader(pdb):
    if not os.path.isfile(pdb):
        cmd.fetch(pdb, path=PDB_DOWNLOAD_PATH)
    else:
        cmd.load(pdb)

def sphere(selection='all', padding=5.0, npts=100):
    coords = cmd.get_coords(selection)
    
    # Calculate the center of the selection
    center = np.mean(coords, axis=0)
    
    # Calculate the maximum distance from the center to any atom in the selection
    max_dist = np.max(np.linalg.norm(coords - center, axis=1))
    
    # Define the radius of the encompassing sphere
    sphere_radius = max_dist + padding
    
    # Generate npts points approximately equally spaced on a sphere (Fibonacci sphere method)
    indices = np.arange(0, npts, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / npts)  # inclination angle
    theta = np.pi * (1 + 5**0.5) * indices    # azimuthal angle
    
    x = sphere_radius * np.cos(theta) * np.sin(phi) + center[0]
    y = sphere_radius * np.sin(theta) * np.sin(phi) + center[1]
    z = sphere_radius * np.cos(phi) + center[2]
    
    sphere_points = np.column_stack([x, y, z])
    return sphere_points
