#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-04-03

# Hungarian Point Set Registration

import typer
from scipy.spatial import cKDTree
import numpy as np
import scipy.spatial.distance as scidist
from scipy.optimize import linear_sum_assignment

# import IPython  # you can use IPython.embed() to explore variables and explore where it's called

app = typer.Typer(
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,  # do not show local variable
    add_completion=False,
)

@app.callback()
def callback(debug:bool=False):
    """
    This is a template file for a Python script using Typer.
    It contains a main function and a test function.
    """
    global DEBUG
    DEBUG = debug
    app.pretty_exceptions_show_locals = debug

def metric(v1, v2, threshold=1e-4):
    """
    >>> v1 = np.asarray([0, 1, 2])
    >>> v2 = np.asarray([1, 0, 2])
    >>> d = metric(v1, v2)
    >>> d
    np.float64(0.0)
    >>> v1 = np.asarray([0, 1, 2])
    >>> v2 = np.asarray([3, 0, 2])
    >>> d = metric(v1, v2)
    >>> d
    np.float64(0.3333333333333333)
    """
    cdist = scidist.cdist(v1[:, None], v2[:, None], metric="euclidean")
    cdist = 1. - (cdist<threshold)
    # print(cdist)
    # row_ind, col_ind = linear_sum_assignment(cdist)
    # d = 1. - (cdist[row_ind, col_ind]).sum()/min(len(row_ind), len(col_ind))
    # d = cdist[row_ind, col_ind].mean()
    d = cdist.min(axis=1).mean()
    # d = 1. - np.isclose(cdist[row_ind, col_ind], 0).sum()/min(len(row_ind), len(col_ind))
    return d


def PSR(coords1, coords2, n_neighbors=8):
    """
    >>> from pymol import cmd
    >>> cmd.fetch("1ycr")
    '1ycr'
    >>> coords1 = cmd.get_coords("chain A")
    >>> coords2 = cmd.get_coords("chain A and resi 50-60+70-75")
    >>> row_ind, col_ind, error = PSR(coords1, coords2+100)
    >>> rmsd = ((coords1[row_ind] - coords2[col_ind])**2).sum(axis=1).mean()
    >>> rmsd
    np.float32(0.0)
    """
    coords1_c = coords1 - coords1.mean(axis=0)
    coords2_c = coords2 - coords2.mean(axis=0)
    tree1 = cKDTree(coords1_c)
    tree2 = cKDTree(coords2_c)
    n_neighbors = min(coords1_c.shape[0], coords2_c.shape[0], n_neighbors)
    distances1, inds1 = tree1.query(coords1_c, k=n_neighbors)
    distances2, inds2 = tree2.query(coords2_c, k=n_neighbors)
    pdist = scidist.cdist(distances1, distances2, metric=metric)
    row_ind, col_ind = linear_sum_assignment(pdist)
    error = pdist[row_ind, col_ind].mean()
    return row_ind, col_ind, error

if __name__ == "__main__":
    import doctest
    import sys

    @app.command()
    def test():
        """
        Test the code
        """
        doctest.testmod(
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF
        )

    @app.command()
    def test_func(func:str):
        """
        Test the given function
        """
        print(f"Testing {func}")
        f = getattr(sys.modules[__name__], func)
        doctest.run_docstring_examples(
            f,
            globals(),
            optionflags=doctest.ELLIPSIS \
                        | doctest.REPORT_ONLY_FIRST_FAILURE \
                        | doctest.REPORT_NDIFF,
        )

    app()

