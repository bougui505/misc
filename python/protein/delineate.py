#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Sep 17 09:27:54 2025

import typer
from pymol import cmd
import gzip
import os
import numpy as np
from PIL import Image
from scipy import ndimage

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
    help="Delineate protein interfaces and visualize them on a surface."
)

def loader(pdb):
    """
    Load a PDB file into PyMOL. Fetches from PDB if not a local file.

    Args:
        pdb (str): Path to PDB file or PDB ID.
    """
    if os.path.isfile(pdb):
        cmd.load(pdb)
    else:
        cmd.fetch(pdb, path=os.path.expanduser("~/pdb"))

def delineate(selection, linewidth=3, color=[0, 255, 0], fill=None, alpha=0.5, filter_patch=True):
    """
    Delineate the contour of a selected molecular surface in PyMOL.

    Args:
        selection (str): PyMOL selection string for the interface.
        linewidth (int): Width of the contour line in pixels.
        color (list): RGB color list for the contour (e.g., [0, 255, 0] for green).
        fill (list | None): RGB color list for filling the delineated interface (e.g., [0, 0, 255] for blue).
                            If None, no fill is applied.
        alpha (float): Transparency of the filled area, between 0.0 (fully transparent) and 1.0 (fully opaque).

    Returns:
        Image.Image: PIL Image object of the contour.
    """
    cmd.create("interface", selection)
    cmd.hide("everything")
    cmd.show_as("surface", "interface")
    cmd.color("black")
    set_view(VIEW)
    cmd.png("tmp/interface.png", width=WIDTH, height=HEIGHT)
    img = np.array(Image.open("tmp/interface.png"))  # (480, 640, 4)
    img = img.sum(axis=2)
    if filter_patch:
        labeled_img, num_features = ndimage.label(img)  # type: ignore
        largest_label = np.argmax([(labeled_img==l).sum() for l in range(1, num_features+1)]) + 1
        sel = (labeled_img == largest_label)
    else:
        sel = (img != 0)
    sel_dil = ndimage.binary_dilation(sel, iterations=linewidth)
    contour = np.int_(sel_dil) - np.int_(sel)
    # plt.matshow(contour)
    # plt.show()
    contour_rgb = np.stack([contour]*3, axis=-1) * np.asarray(color)  # save it in green (RGB mode)
    if fill is not None:
        print(f"Filling with color: {fill}")
        filling = np.stack([np.int_(sel)]*3, axis=-1) * np.asarray(fill)
        contour_rgb += filling
    else:
        filling = None
    contour_img = Image.fromarray(np.uint8(contour_rgb), 'RGB')
    alpha_arr = np.int_(contour_rgb.sum(axis=-1)!=0) * 255
    if filling is not None:
        alpha_arr += np.int_(filling.sum(axis=-1)!=0) * int(255*alpha)
    contour_img.putalpha(Image.fromarray(np.uint8(alpha_arr)))
    cmd.disable("interface")
    return contour_img

def set_view(view):
    """
    Set the PyMOL camera view using a view matrix string.

    Args:
        view (str or None): Comma-separated string of 16 view matrix values representing
                            the PyMOL camera orientation and position (e.g., '1,0,0,0,0,1,0,0,0,0,1,0,100,200,300,1').
                            If None, PyMOL's current default view is used.
    """
    if view is not None:
        view_mat = view.strip().split(",")
        cmd.set_view(view_mat)

@app.command(
    help="Delineate a protein interface and visualize it on a surface. "
         "Loads a PDB, delineates a specified interface with a contour and optional fill, "
         "and renders it onto the surface of a reference selection. "
         "The reference selection chains are colored with a grayscale gradient."
)
def main(
        pdb: str = typer.Argument(..., help="Path to the PDB file or PDB ID to load into PyMOL."),
        ref: str = typer.Argument(..., help="PyMOL selection string for the reference structure whose surface will be displayed. Chains in this selection will be colored with a grayscale gradient (e.g., 'chain A or chain B')."),
        sel: str = typer.Argument(..., help="PyMOL selection string for the interface region to be delineated. This selection defines the area on the surface where the contour will be drawn (e.g., 'chain A and around 5 of chain B')."),
        outfile: str = typer.Argument(..., help="Output filename for the generated image (e.g., 'interface.png')."),
        infile: str|None = typer.Option(None, help="Optional: Input filename for a pre-rendered surface image. If provided, the delineated contour will be overlaid on this image instead of rendering a new surface from PyMOL. (e.g., 'pre_rendered_surface.png')"),
        color: str = typer.Option('255,0,0', help="RGB color (comma-separated, 0-255) for the contour line. Example: '255,0,0' for red."),
        linewidth: int = typer.Option(3, min=1, help="Width of the contour line in pixels."),
        fill: str | None = typer.Option(None, help="RGB color (comma-separated, 0-255) for filling the delineated interface region. Example: '0,0,255' for blue. If None, only the contour line will be drawn."),
        alpha: float = typer.Option(0.5, min=0.0, max=1.0, help="Transparency of the filled area, between 0.0 (fully transparent) and 1.0 (fully opaque)."),
        view: str | None = typer.Option(None, help="Comma-separated string of 16 view matrix values for PyMOL camera (e.g., '1,0,0,0,0,1,0,0,0,0,1,0,100,200,300,1'). If None, PyMOL's default view is used."),
        debug: bool = typer.Option(False, help="If True, enables debug mode which provides more detailed error messages by showing local variables on exceptions."),
        width: int = typer.Option(2560, min=1, help="Width of the generated output image in pixels."),
        height: int = typer.Option(1920, min=1, help="Height of the generated output image in pixels."),
        tmpdir: str = typer.Option("tmp", help="Directory to store temporary PyMOL PNGs and other intermediate files. The directory will be created if it doesn't exist."),
        filter_patch: bool = typer.Option(True, help="If True, only the largest connected component of the delineated interface will be kept (removes small patches)."),
    ):
    """
    Delineate a protein interface and visualize it on a surface.

    This command loads a PDB structure, defines a reference surface (e.g., the whole protein),
    and a specific interface area. It then calculates and draws a contour line
    around the interface on the reference surface. The output is an image
    showing the contoured interface.

    Args:
        pdb (str): Path to the PDB file or PDB ID to load into PyMOL.
        ref (str): PyMOL selection string for the reference structure whose surface
                   will be displayed. Chains in this selection will be colored
                   with a grayscale gradient.
        sel (str): PyMOL selection string for the interface region to be delineated.
                   This selection defines the area on the surface where the contour
                   will be drawn.
        outfile (str): Output filename for the generated image.
        infile (str | None): Optional: Input filename for a pre-rendered surface image.
                             If provided, the delineated contour will be overlaid on this
                             image instead of rendering a new surface from PyMOL.
        color (str): Comma-separated RGB values (0-255) for the contour line color.
                     Example: '255,0,0' for red.
        linewidth (int): The thickness of the contour line in pixels in the output image.
        fill (str | None): Comma-separated RGB values (0-255) for filling the delineated
                           interface region. Example: '0,0,255' for blue. If not provided,
                           only the contour line will be drawn.
        alpha (float): Transparency of the filled area, between 0.0 (fully transparent)
                       and 1.0 (fully opaque).
        view (str | None): A string containing 16 comma-separated floats representing
                           the PyMOL view matrix. This allows for precise camera positioning.
                           If not specified, PyMOL's current view will be used.
        debug (bool): If True, enables debug mode which provides more detailed
                      error messages by showing local variables on exceptions.
        width (int): The width of the generated output image in pixels.
        height (int): The height of the generated output image in pixels.
        tmpdir (str): The directory where temporary PyMOL PNGs and the final
                      image will be saved. The directory will be created if it doesn't exist.
        filter_patch (bool): If True, only the largest connected component of the
                             delineated interface will be kept (removes small patches).
    """
    global DEBUG, WIDTH, HEIGHT, VIEW

    DEBUG = debug
    app.pretty_exceptions_show_locals = debug
    WIDTH = width
    HEIGHT = height
    os.makedirs(tmpdir, exist_ok=True)
    loader(pdb)
    global VIEW
    VIEW = view
    color_rgb = [int(_.strip()) for _ in color.split(",")]
    if fill is not None:
        fill_rgb = [int(_.strip()) for _ in fill.split(",")]
    else:
        fill_rgb = None
    contour_img =  delineate(sel, color=color_rgb, fill=fill_rgb, linewidth=linewidth, alpha=alpha, filter_patch=filter_patch)
    # get a list of all the chains in ref selection
    cmd.create("ref", ref)
    chains_ref = cmd.get_chains("ref")
    print(f"{chains_ref=}")
    num_chains = len(chains_ref)
    for i, chain_id in enumerate(chains_ref):
        # Calculate a grayscale value (from 0 for black to 1 for white)
        gray_value = i+1 / (num_chains - 1) if num_chains > 1 else 0.5
        color_name = f"gray_chain_{chain_id}"
        cmd.set_color(color_name, [gray_value*255, gray_value*255, gray_value*255])
        cmd.color(color_name, f"chain {chain_id} and ref")
    cmd.show_as("surface", "ref")
    set_view(VIEW)
    if infile is None:
        cmd.png(f"{tmpdir}/surface.png", width=WIDTH, height=HEIGHT)
        surface_img = Image.open(f"{tmpdir}/surface.png")
    else:
        surface_img = Image.open(infile)
    # stack tmp/contour.png on top of the surface
    surface_img.paste(contour_img, (0, 0), contour_img)
    surface_img.save(outfile)

if __name__ == "__main__":
    import doctest
    import sys
    app()

