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
import numpy as np
import scipy.spatial.distance as scidist
from tqdm import tqdm

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

def isin_tolerance(dmat_big, dmat_small, tol=1e-4):
    # Reshape dmat_big to (10, 10, 1, 1) and dmat_small to (4, 4)
    # This allows broadcasting to a (10, 10, 4, 4) comparison matrix
    diffs = np.abs(dmat_big[:, :, np.newaxis, np.newaxis] - dmat_small)
    
    # Check where the absolute difference is within tolerance
    in_tolerance = np.any(diffs <= tol, axis=(2, 3))
    # in_tolerance = diffs <= tol
    
    return in_tolerance

def rigid_body_fit(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Numpy array of shape (N,D) -- Point Cloud to Align (source)
        -    B: Numpy array of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.dot(B_c)
    U, S, Vt = np.linalg.svd(H)
    V = Vt.T
    # Rotation matrix calculation with reflection check
    d = np.linalg.det(V.dot(U.T))
    if d < 0:
        V[:, -1] *= -1
    # Rotation matrix
    R = V.dot(U.T)
    # Translation vector
    t = b_mean - R.dot(a_mean)
    return R, t

def PSR(A, B, tol=1e-4):
    """
    >>> from scipy.spatial.transform import Rotation as R
    >>> from pymol import cmd
    >>> cmd.fetch("1ycr")
    '1ycr'
    >>> coords1 = cmd.get_coords("chain A and resi 50-60+70-75")
    >>> coords2 = cmd.get_coords("chain A")

    >>> rot = R.from_euler('zx', [90, 45], degrees=True)
    >>> coords1 = rot.apply(coords1) + 100.
    >>> coords1_back = cmd.get_coords("chain A and resi 50-60+70-75")

    >>> i,j,rmsd = PSR(coords1, coords2)
    >>> rmsd
    2.358980942620816e-06
    """
    if A.shape[0] > B.shape[0]:
        big = A
        small = B
    else:
        big = B
        small = A
    n_small = small.shape[0]
    n_big = big.shape[0]
    dmat_big = scidist.squareform(scidist.pdist(big))
    dmat_small = scidist.squareform(scidist.pdist(small))
    small_ind = []
    big_ind = []
    bigset = set(np.arange(n_big))
    nmatch = 0
    with tqdm(dmat_small) as pbar:
        for ismall, vsmall in enumerate(pbar):
            # for ibig, vbig in enumerate(dmat_big):
            for ibig in bigset:
                vbig = dmat_big[ibig]
                dmat_i = scidist.cdist(vsmall[:,None],vbig[:,None])
                r = (dmat_i<tol).sum()/n_small
                if r>=1:  # all the distances match
                    small_ind.append(ismall)
                    big_ind.append(ibig)
                    bigset -= {ibig}
                    nmatch+=1
                    pbar.set_postfix(matches=f"{nmatch}/{n_small}")
                    break
    R, t = rigid_body_fit(small[small_ind], big[big_ind])
    small_aligned = (R.dot(small.T)).T + t
    rmsd = np.sqrt(((small_aligned - big[big_ind])**2).sum(axis=1).mean())
    return small_ind, big_ind, float(rmsd)



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

