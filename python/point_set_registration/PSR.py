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
import os
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed
import multiprocessing

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

def _match_profiles(vsmall, vbig):
    dmat_i = scidist.cdist(vsmall[:, None], vbig[:, None])
    row_ind, col_ind = linear_sum_assignment(dmat_i)
    return np.median(dmat_i[row_ind, col_ind])

def get_consistency_inliers(S, B, tol=2.0):
    """
    Find the largest subset of matches (S, B) that are internally consistent
    in terms of pairwise distances.
    """
    if len(S) < 3:
        return np.ones(len(S), dtype=bool)
    D_S = scidist.squareform(scidist.pdist(S))
    D_B = scidist.squareform(scidist.pdist(B))
    Diff = np.abs(D_S - D_B)
    consistency = (Diff < tol).sum(axis=1)
    seed_idx = np.argmax(consistency)
    inliers = Diff[seed_idx] < tol
    return inliers

def ransac_seed(S, B, tol=2.0, n_iterations=1000, n_seeds=1, verbose=False):
    """
    Find an initial rigid body transformation using RANSAC on triplets.
    """
    n_matches = len(S)
    if n_matches < 3:
        R, t = rigid_body_fit(S, B)
        if n_seeds > 1:
            return [(R, t, n_matches)]
        return R, t
    
    D_S = scidist.squareform(scidist.pdist(S))
    D_B = scidist.squareform(scidist.pdist(B))
    Diff = np.abs(D_S - D_B)
    
    seeds = [] # List of (R, t, n_inliers)
    
    for _ in range(n_iterations):
        # We take randomly an element of D_S
        idx1 = np.random.randint(n_matches)
        # We check for neighbors in D_B (dist<tol)
        consistent_with_1 = np.where(Diff[idx1] < tol)[0]
        if len(consistent_with_1) < 3:
            continue
        
        # then we choose randomly 3 of this neighbors
        sample_idx = np.random.choice(consistent_with_1, size=3, replace=False)
        # to compute an alignment between S and B
        R_seed, t_seed = rigid_body_fit(S[sample_idx], B[sample_idx])
        
        # Check inliers among the provided matches
        S_aligned = (R_seed.dot(S.T)).T + t_seed
        # then we check the new distances between S_aligned and B
        dists = np.linalg.norm(S_aligned - B, axis=1)
        inliers = dists < 5.0 # Loose tolerance for initial seed
        n_inliers = inliers.sum()
        
        seeds.append((R_seed, t_seed, n_inliers))
            
    # Sort seeds by number of inliers
    seeds = sorted(seeds, key=lambda x: x[2], reverse=True)
    
    if n_seeds > 1:
        return seeds[:n_seeds]
    
    if len(seeds) > 0:
        best_R, best_t, best_n_inliers = seeds[0]
        if verbose:
            print(f"RANSAC seed found {best_n_inliers}/{n_matches} inliers")
        return best_R, best_t
    else:
        return rigid_body_fit(S, B)

def _refine_seed(seed, small, big, tol, verbose, seed_idx, n_seeds, small_ind, big_ind):
    R_seed, t_seed, _ = seed
    if verbose and n_seeds > 1:
        print(f"Refining from RANSAC seed {seed_idx+1}/{n_seeds}...")
    small_aligned = (R_seed.dot(small.T)).T + t_seed

    assignments, rmsd = set(), -1
    curr_small_ind, curr_big_ind = small_ind, big_ind
    R, t = R_seed, t_seed
    while True:
        # Re-assign based on Euclidean distance
        dmat = scidist.cdist(small_aligned, big)
        small_ind_ref, big_ind_ref = linear_sum_assignment(dmat)
        
        # Distance thresholding for outlier rejection
        dists = dmat[small_ind_ref, big_ind_ref]
        mask = dists < 5.0
        small_ind_ref, big_ind_ref = small_ind_ref[mask], big_ind_ref[mask]
        
        if len(small_ind_ref) < 3:
            break

        # Identify the consistent inliers for the rigid fit
        inliers = get_consistency_inliers(small[small_ind_ref], big[big_ind_ref], tol=tol)
        curr_small_ind, curr_big_ind = small_ind_ref[inliers], big_ind_ref[inliers]

        # Termination condition based on the set of matched indices
        assignment = (tuple(curr_small_ind), tuple(curr_big_ind))
        if assignment in assignments:
            break
        assignments.add(assignment)
        
        # Update the rigid body transformation
        if len(curr_small_ind) >= 3:
            R, t = rigid_body_fit(small[curr_small_ind], big[curr_big_ind])
        else:
            # Fallback to all matches if consistency fails
            R, t = rigid_body_fit(small[small_ind_ref], big[big_ind_ref])
        small_aligned = (R.dot(small.T)).T + t

        rmsd = np.sqrt(((small_aligned[curr_small_ind] - big[curr_big_ind])**2).sum(axis=1).mean())
        if verbose and n_seeds == 1:
            print(f"Iterative refined RMSD: {rmsd:.4f} (on {len(curr_small_ind)} inliers)")
        
    return (curr_small_ind, curr_big_ind, float(rmsd), R, t)

def PSR(A, B, n_jobs=None, n_iterations=2000, n_seeds=5, verbose=False):
    """
    >>> from scipy.spatial.transform import Rotation as R
    >>> from pymol import cmd
    >>> cmd.fetch("1ycr")
    '1ycr'
    >>> coords1 = cmd.get_coords("chain A and resi 50-60+70-75")
    >>> coords2 = cmd.get_coords("chain A")

    # shuffle the order of coords2
    >>> coords2 = coords2[np.random.choice(coords2.shape[0], replace=False, size=coords2.shape[0]), :]

    >>> rot = R.from_euler('zx', [90, 45], degrees=True)
    >>> coords1 = rot.apply(coords1) + 100.

    # # Add noise
    # >>> coords1 += np.random.uniform(0., 0.1, size=coords1.shape)

    >>> i,j,rmsd,R,t = PSR(coords1, coords2)
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
    tol = 2.0 # Tolerance for geometric consistency check
    if n_jobs is None:
        n_jobs = multiprocessing.cpu_count()
    
    # Precompute cost matrix in parallel
    if verbose:
        print(f"Computing cost matrix ({n_small}x{n_big}) using {n_jobs} cores...")
    
    costs = Parallel(n_jobs=n_jobs)(
        delayed(_match_profiles)(dmat_small[i], dmat_big[j])
        for i in tqdm(range(n_small), desc="Distance profiles", disable=not verbose)
        for j in range(n_big)
    )
    cost_matrix = np.array(costs).reshape(n_small, n_big)
    
    # Global assignment
    if verbose:
        print("Solving global assignment...")
    small_ind, big_ind = linear_sum_assignment(cost_matrix)
    
    # Robust seeding using RANSAC to handle dimers/symmetries
    n_iterations = max(n_seeds, n_iterations)
    seeds = ransac_seed(small[small_ind], big[big_ind], tol=tol, n_iterations=n_iterations, n_seeds=n_seeds, verbose=verbose)
    if isinstance(seeds, tuple): # Only one seed returned
        seeds = [seeds + (None,)]

    # Parallel refinement
    if verbose:
        print(f"Refining {len(seeds)} RANSAC seeds using {n_jobs} cores...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(_refine_seed)(
            seed, small, big, tol, False, i, len(seeds), small_ind, big_ind
        ) for i, seed in enumerate(tqdm(seeds))
    )

    best_res = None
    best_rmsd = np.inf
    for res in results:
        rmsd = res[2]
        if rmsd != -1 and rmsd < best_rmsd:
            best_rmsd = rmsd
            best_res = res
            
    return best_res

@app.command()
def fit(
    pdb1: str = typer.Argument(..., help="First PDB file (local path or PDB ID)"),
    pdb2: str = typer.Argument(..., help="Second PDB file (local path or PDB ID)"),
    sel1: str = typer.Option("all", help="Selection string for the first PDB"),
    sel2: str = typer.Option("all", help="Selection string for the second PDB"),
    n_iterations: int = typer.Option(1000, help="Number of RANSAC iterations"),
    n_seeds: int = typer.Option(3, help="Number of RANSAC seeds to refine"),
    n_jobs: int = typer.Option(None, help="Number of jobs for parallel processing"),
    verbose: bool = typer.Option(True, help="Show progress messages")
):
    """
    Perform point set registration between two PDB files.
    """
    from pymol import cmd
    if os.path.exists(pdb1):
        cmd.load(pdb1, "pdb1")
    else:
        cmd.fetch(pdb1, "pdb1")
    if os.path.exists(pdb2):
        cmd.load(pdb2, "pdb2")
    else:
        cmd.fetch(pdb2, "pdb2")
    coords1 = cmd.get_coords(f"pdb1 and ({sel1})")
    coords2 = cmd.get_coords(f"pdb2 and ({sel2})")
    n1 = coords1.shape[0]
    n2 = coords2.shape[0]
    small_ind, big_ind, rmsd, R, t = PSR(coords1, coords2, n_jobs=n_jobs, n_iterations=n_iterations, n_seeds=n_seeds, verbose=verbose)
    print(f"{rmsd=}")
    if n1 > n2:
        big = coords1
        small = coords2
        to_align = pdb2
        obj = "pdb2"
    else:
        big = coords2
        small = coords1
        to_align = pdb1
        obj = "pdb1"
    aligned = (R.dot(cmd.get_coords(obj).T)).T + t
    cmd.load_coords(aligned, obj, 1)
    cmd.save(f"{to_align}_aligned.pdb", obj)


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

