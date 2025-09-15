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
import scipy.spatial.distance as scidist

import typer

app = typer.Typer()

PDB_DOWNLOAD_PATH = os.path.expanduser("~/pdb")

def loader(pdb):
    """
    Load a PDB structure into PyMOL. Fetches from PDB if not a local file.
    """
    print(f"Loading {pdb}")
    if not os.path.isfile(pdb):
        cmd.fetch(pdb, path=PDB_DOWNLOAD_PATH)
    else:
        cmd.load(pdb)
    cmd.orient()

def sphere(selection='all', padding=5.0, npts=100):
    """
    Generate points on a sphere encompassing the given selection.

    Args:
        selection (str): PyMOL selection string.
        padding (float): Extra padding for the sphere radius.
        npts (int): Number of points to generate on the sphere.

    Returns:
        np.ndarray: An array of shape (npts, 3) with the coordinates of the sphere points.
    """
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

def show_sphere(selection='all', padding=5.0, npts=100):
    """
    Visualize the encompassing sphere points as pseudoatoms in PyMOL.

    Args:
        selection (str): PyMOL selection string.
        padding (float): Extra padding for the sphere radius.
        npts (int): Number of points to generate on the sphere.
    """
    sphere_points = sphere(selection=selection, padding=padding, npts=npts)
    for i, coords in enumerate(sphere_points):
        cmd.pseudoatom(object="sphere", pos=list(coords))
    cmd.show_as("spheres", "sphere")
    cmd.orient()

class Labeler():
    """
    A class to manage and place labels around a selection in PyMOL using a sphere of points.
    """
    def __init__(self, selection="all", padding=2.0, npts=100, linecolor="black"):
        """
        Initializes the Labeler with an encompassing sphere of points.

        Args:
            selection (str): PyMOL selection string to define the sphere's center and radius.
            padding (float): Extra padding for the sphere radius.
            npts (int): Number of points to generate on the sphere.
        """
        self.sphere_points = sphere(selection=selection, padding=padding, npts=npts)
        self.labelid = 0
        cmd.set("dash_color", linecolor)  # type: ignore

    def label(self, selection, labelname):
        """
        Places a label on the sphere closest to the given selection.
        Removes the chosen sphere point to avoid duplicate labels at the same location.

        Args:
            selection (str): PyMOL selection string for the atom to label.
                             Must select exactly one atom.
            labelname (str): The text content of the label.
        """
        coords = cmd.get_coords(selection)
        assert coords.shape[0] == 1, "Only 1 atom must be selected for labelling"
        dmat = scidist.cdist(coords, self.sphere_points).squeeze()  # shape (npts,)
        ptid = dmat.argmin()
        labelcoords = self.sphere_points[ptid]
        self.sphere_points = np.delete(self.sphere_points, (ptid), axis=0)
        labelid = f"l_{labelname.replace(" ", "_")}_{self.labelid}"
        self.labelid += 1
        cmd.pseudoatom(object=labelid, pos=list(labelcoords))
        cmd.label(labelid, f"'{labelname}'")
        cmd.distance(f"d_{labelid}", selection, labelid)
        cmd.hide("labels", f"d_{labelid}")

@app.command()
def main():
    """
    Main function to demonstrate the captioning functionality.
    Loads a PDB, initializes a Labeler, and adds example labels.
    """
    loader("1t4e_A")
    # show_sphere()
    labeler = Labeler()
    labeler.label('resi 59 and name CA', "resi 59")
    labeler.label('resi 59 and name CA', "resi 59_2")

if __name__ == "__main__":
    app()
