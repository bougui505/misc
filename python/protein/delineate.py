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
)

def loader(pdb):
    if os.path.isfile(pdb):
        cmd.load(pdb)
    else:
        cmd.fetch(pdb, path=os.path.expanduser("~/pdb"))

def delineate(selection, linewidth=3, color=[0, 255, 0]):
    cmd.create("interface", selection)
    set_view(VIEW)
    cmd.hide("everything")
    cmd.show_as("surface", "interface")
    cmd.color("black")
    cmd.png("tmp/interface.png", width=WIDTH, height=HEIGHT)
    img = np.array(Image.open("tmp/interface.png"))  # (480, 640, 4)
    img = img.sum(axis=2)
    labeled_img, num_features = ndimage.label(img)  # type: ignore
    largest_label = np.argmax([(labeled_img==l).sum() for l in range(1, num_features+1)]) + 1
    sel = (labeled_img == largest_label)
    # sel = (img != 0)
    sel_dil = ndimage.binary_dilation(sel, iterations=linewidth)
    contour = np.int_(sel_dil) - np.int_(sel)
    # plt.matshow(contour)
    # plt.show()
    contour_rgb = np.stack([contour]*3, axis=-1) * np.asarray(color)  # save it in green (RGB mode)
    contour_img = Image.fromarray(np.uint8(contour_rgb), 'RGB')
    contour_img.putalpha(Image.fromarray(np.uint8(contour * 255)))
    contour_img.save("tmp/contour.png")
    cmd.disable("interface")
    return contour_img
    # stack tmp/contour.png on top of figures/footprints.png
    # surface_img = Image.open("figures/footprints.png")
    # surface_img.paste(contour_img, (0, 0), contour_img)
    # surface_img.save("figures/footprints.png")

def set_view(view):
    if view is not None:
        view_mat = view.strip().split(",")
        cmd.set_view(view_mat)

@app.command()
def main(
        pdb:str,
        ref:str,
        sel:str,
        view:str|None=None,
        debug:bool=False,
        width:int=2560,
        height:int=1920,
        tmpdir:str="tmp",
    ):
    """"""
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = debug
    global WIDTH
    global HEIGHT
    WIDTH = width
    HEIGHT = height
    os.makedirs(tmpdir, exist_ok=True)
    loader(pdb)
    global VIEW
    VIEW = view
    set_view(VIEW)
    contour_img =  delineate(sel)
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
    cmd.png(f"{tmpdir}/surface.png", width=WIDTH, height=HEIGHT)
    # stack tmp/contour.png on top of the surface
    surface_img = Image.open(f"{tmpdir}/surface.png")
    surface_img.paste(contour_img, (0, 0), contour_img)
    surface_img.save(f"{tmpdir}/footprints.png")

if __name__ == "__main__":
    import doctest
    import sys
    app()

