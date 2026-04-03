#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-02 10:00:15 (UTC+0200)

import sys
import os
import scipy.optimize
import numpy as np
import torch
import pymol
from pymol import cmd


def print_progress(instr):
    sys.stdout.write(f'{instr}\r')
    sys.stdout.flush()


def find_rigid_alignment(A, B):
    """
    See: https://en.wikipedia.org/wiki/Kabsch_algorithm
    2-D or 3-D registration with known correspondences.
    Registration occurs in the zero centered coordinate system, and then
    must be transported back.
        Args:
        -    A: Torch tensor of shape (N,D) -- Point Cloud to Align (source)
        -    B: Torch tensor of shape (N,D) -- Reference Point Cloud (target)
        Returns:
        -    R: optimal rotation
        -    t: optimal translation
    Test on rotation + translation and on rotation + translation + reflection
        >>> A = torch.tensor([[1., 1.], [2., 2.], [1.5, 3.]], dtype=torch.float)
        >>> R0 = torch.tensor([[np.cos(60), -np.sin(60)], [np.sin(60), np.cos(60)]], dtype=torch.float)
        >>> B = (R0.mm(A.T)).T
        >>> t0 = torch.tensor([3., 3.])
        >>> B += t0
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
        >>> B *= torch.tensor([-1., 1.])
        >>> R, t = find_rigid_alignment(A, B)
        >>> A_aligned = (R.mm(A.T)).T + t
        >>> rmsd = torch.sqrt(((A_aligned - B)**2).sum(axis=1).mean())
        >>> rmsd
        tensor(3.7064e-07)
    """
    a_mean = A.mean(axis=0)
    b_mean = B.mean(axis=0)
    A_c = A - a_mean
    B_c = B - b_mean
    # Covariance matrix
    H = A_c.T.mm(B_c)
    U, S, V = torch.svd(H)
    # Rotation matrix
    R = V.mm(U.T)
    # Translation vector
    t = b_mean[None, :] - R.mm(a_mean[None, :].T).T
    t = t.T
    return R, t.squeeze()


def transform(coords, R, t):
    """
    Apply R and a translation t to coords
    """
    coords_out = R.mm(coords.T).T + t
    return coords_out


def get_RMSD(A, B):
    """
    Return the RMSD between the two set of coords
    """
    rmsd = torch.sqrt(((A - B)**2).sum(axis=1).mean())
    return rmsd


def get_topology(resids, chains):
    """
    Return the adjacency matrix encoding the topology of the object
    - resids: list of resid of the anchors
    - chains: list of chains per CA of the anchors
    """
    n = len(resids)
    adjmat = np.zeros((n, n))
    for i in range(1, n):
        dr = resids[i] - resids[i - 1]
        cr = chains[i] == chains[i - 1]
        if dr == 1 and cr:
            adjmat[i, i - 1] = 1.
            adjmat[i - 1, i] = 1.
    return adjmat


def assign_anchors(coords, coords_ref, dist_thr=None, return_perm=False, cdist=None):
    """
    Assign the closest anchors with coords coords_ref
    - cdist: precomputed distance matrix between coords and coords_ref

    # Test with equal shape for coords and coords_ref
    >>> coords_ref = torch.tensor([[0., 0., 0.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> coords = torch.zeros_like(coords_ref)
    >>> coords[0] = coords_ref[1]
    >>> coords[1] = coords_ref[3]
    >>> coords[2] = coords_ref[0]
    >>> coords[3] = coords_ref[2]
    >>> coords
    tensor([[1., 2., 3.],
            [7., 8., 9.],
            [0., 0., 0.],
            [4., 5., 6.]])
    >>> get_RMSD(coords, coords_ref)
    tensor(7.5166)
    >>> assignment, sel, P = assign_anchors(coords, coords_ref, return_perm=True)
    >>> coords_ordered = coords[assignment]
    >>> (coords.T.mm(P).T == coords_ordered).all()
    tensor(True)
    >>> coords_ordered
    tensor([[0., 0., 0.],
            [1., 2., 3.],
            [4., 5., 6.],
            [7., 8., 9.]])
    >>> get_RMSD(coords_ordered, coords_ref[sel])
    tensor(0.)


    # Test with size of coords_ref lower than coords
    >>> coords_ref = torch.tensor([[-1., -2., -3.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    >>> coords = torch.zeros((5, 3))
    >>> coords[0] = coords_ref[1]
    >>> coords[1] = coords_ref[3]
    >>> coords[4] = coords_ref[0]
    >>> coords[3] = coords_ref[2]
    >>> coords[2] = torch.tensor([10., 11., 12.])
    >>> coords
    tensor([[ 1.,  2.,  3.],
            [ 7.,  8.,  9.],
            [10., 11., 12.],
            [ 4.,  5.,  6.],
            [-1., -2., -3.]])
    >>> assignment, sel, P = assign_anchors(coords, coords_ref, return_perm=True)
    >>> coords_ordered = coords[assignment]
    >>> (coords.T.mm(P).T == coords_ordered).all()
    tensor(True)
    >>> coords_ordered
    tensor([[-1., -2., -3.],
            [ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.]])
    >>> get_RMSD(coords_ordered, coords_ref[sel])
    tensor(0.)

    # Test with size of coords_ref higher than coords
    >>> coords_ref = torch.tensor([[-1., -2., -3.], [1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10, 11, 12]])
    >>> coords = torch.zeros((4, 3))
    >>> coords[0] = coords_ref[1]
    >>> coords[1] = coords_ref[3]
    >>> coords[2] = coords_ref[0]
    >>> coords[3] = coords_ref[4]
    >>> coords
    tensor([[ 1.,  2.,  3.],
            [ 7.,  8.,  9.],
            [-1., -2., -3.],
            [10., 11., 12.]])
    >>> assignment, sel, P = assign_anchors(coords, coords_ref, return_perm=True)
    >>> coords_ordered = coords[assignment]
    >>> (coords.T.mm(P).T == coords_ordered).all()
    tensor(True)
    >>> coords_ordered
    tensor([[-1., -2., -3.],
            [ 1.,  2.,  3.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])
    >>> sel
    array([0, 1, 3, 4])
    >>> assignment
    array([2, 0, 1, 3])
    >>> get_RMSD(coords_ordered, coords_ref[sel])
    tensor(0.)

    >>> coords[2] += 100.
    >>> assignment, sel, P = assign_anchors(coords, coords_ref, dist_thr=4., return_perm=True)
    >>> coords_ref
    tensor([[-1., -2., -3.],
            [ 1.,  2.,  3.],
            [ 4.,  5.,  6.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])
    >>> coords
    tensor([[ 1.,  2.,  3.],
            [ 7.,  8.,  9.],
            [99., 98., 97.],
            [10., 11., 12.]])
    >>> coords_ordered = coords[assignment]
    >>> coords_ordered
    tensor([[ 1.,  2.,  3.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])
    >>> get_RMSD(coords_ordered, coords_ref[sel])
    tensor(0.)
    >>> coords.T.mm(P).T
    tensor([[ 1.,  2.,  3.],
            [ 7.,  8.,  9.],
            [10., 11., 12.]])
    """
    if cdist is None:
        cdist = torch.cdist(coords, coords_ref)
        cdist = cdist.cpu().numpy()
        if dist_thr is not None:
            cdist[cdist > dist_thr] = 9999.99
    row_ind, col_ind = scipy.optimize.linear_sum_assignment(cdist)
    if dist_thr is not None:
        distances = cdist[row_ind, col_ind]
        # sel = distances <= dist_thr
        sel = distances < 9999.99
        row_ind = row_ind[sel]
        col_ind = col_ind[sel]
    n = coords.shape[0]
    n_ref = coords_ref.shape[0]
    assignment = -np.ones(n_ref, dtype=int)
    assignment[col_ind] = row_ind
    assignment = assignment[assignment > -1]
    sel = list(col_ind)
    sel.sort()
    sel = np.asarray(sel)
    if return_perm:
        P = torch.zeros((n, n_ref))
        P[assignment, torch.arange(len(assignment))] = 1.
        P = P[:, :n]
        # P[:, P.sum(axis=0) == 0] = 1 / n
        P = P[:, P.sum(axis=0) != 0]
        return assignment, sel, P
    else:
        return assignment, sel


def find_initial_alignment(coords, coords_ref, fsize=30):
    """
    - fsize: fragment size
    """
    chunks = torch.split(coords, fsize)[:-1]
    chunks_ref = torch.split(coords_ref, fsize)[:-1]
    n_chunks = len(chunks)
    rmsd_min = 9999.99
    for i, chunk in enumerate(chunks):
        for j, chunk_ref in enumerate(chunks_ref):
            R, t = find_rigid_alignment(chunk, chunk_ref)
            chunk_aligned = transform(chunk, R, t)
            # coords_aligned = transform(coords, R, t)
            rmsd = get_RMSD(chunk_aligned, chunk_ref)
            # rmsd = get_RMSD(coords_aligned, coords_ref)
            if rmsd < rmsd_min:
                rmsd_min = rmsd
                i_best, j_best = i, j
                R_best, t_best = R, t
    # print(rmsd_min, i_best, j_best)
    return R_best, t_best


def icp(coords, coords_ref, device, maxiter, dist_thr=3.8, lstsq_fit_thr=0., verbose=True, stop=1e-3, return_Rt=False):
    """
    Iterative Closest Point
    - lstsq_fit_thr: distance threshold for least square fit (if 0: no lstsq_fit)
    """
    coords_out = coords.detach().clone()
    R, t = find_initial_alignment(coords_out, coords_ref)
    coords_out = transform(coords_out, R, t)
    assignment, sel = assign_anchors(coords_ref, coords_out, dist_thr=dist_thr)
    rmsd = get_RMSD(coords_ref[assignment], coords_out[sel])
    n_assigned = len(sel)
    if verbose:
        print(f"Initial RMSD: {rmsd} Å; n_assigned: {n_assigned}/{len(coords)} at less than {dist_thr} Å")
    for i in range(maxiter):
        R, t = find_rigid_alignment(coords_out[sel], coords_ref[assignment])
        coords_out = transform(coords_out, R, t)
        assignment, sel = assign_anchors(coords_ref, coords_out, dist_thr=dist_thr)
        rmsd = get_RMSD(coords_out[sel], coords_ref[assignment])
        n_assigned = len(sel)
        if verbose:
            print_progress(
                f'{i+1}/{maxiter}: {rmsd} Å; n_assigned: {n_assigned}/{len(coords)} at less than {dist_thr} Å             '
            )
        if rmsd <= stop:
            break
    if verbose:
        sys.stdout.write('\n')
        print("---")
        print(f"RMSD: {rmsd:.3f}")
    if lstsq_fit_thr > 0.:
        coords_out = lstsq_fit(coords_out, coords_ref, dist_thr=lstsq_fit_thr)
        assignment, sel = assign_anchors(coords_ref, coords_out, dist_thr=dist_thr)
        rmsd = get_RMSD(coords_out[sel], coords_ref[assignment])
        if verbose:
            print(f'lstsq_fit: {rmsd} Å; n_assigned: {n_assigned}/{len(coords)} at less than {dist_thr} Å')
            sys.stdout.write('\n')
    if not return_Rt:
        return coords_out, float(rmsd)
    else:
        return R, t


def lstsq_fit(coords, coords_ref, dist_thr=1.9, ca_dist=3.8):
    """
    Perform a least square fit of coords on coords_ref
    """
    n = coords.shape[0]
    coords_out = torch.clone(coords)
    device = coords_out.device
    coords_out = coords_out.to('cpu')
    assignment, sel = assign_anchors(coords_ref, coords, dist_thr=dist_thr)
    # Not yet implemented on gpu so go to cpu:
    if coords_ref.is_cuda:
        coords_ref = coords_ref.to('cpu')
    if coords.is_cuda:
        coords = coords.to('cpu')
    # Topology
    anchors = coords_ref[assignment]
    pdist = torch.cdist(anchors, anchors)
    sequential = torch.diagonal(pdist, offset=1)
    sigma_ca = 0.1
    topology = torch.exp(-(sequential - ca_dist) / (2 * sigma_ca**2))
    toposel = torch.nonzero(topology > .5, as_tuple=True)[0]
    sel = sel[toposel]
    assignment = assignment[toposel]
    ##########
    if coords[sel].shape[0] > 3:
        X, _ = torch.lstsq(coords_ref[assignment].T, coords[sel].T)
        coords_out[sel] = (coords[sel].T.mm(X[:n])).T
        n_assigned = len(sel)
        print(f"lstsq_fit: n_assigned: {n_assigned}/{n} at less than {dist_thr} Å")
    coords_out = coords_out.to(device)
    return coords_out


def get_resids(obj):
    """
    Return the list of resids for the given obj
    """
    myspace = {'resids': []}
    cmd.iterate(obj, 'resids.append(resi)', space=myspace)
    resids = np.int_(myspace['resids'])
    return resids


def get_sequence(obj):
    aa1 = list("ACDEFGHIKLMNPQRSTVWY")
    aa3 = "ALA CYS ASP GLU PHE GLY HIS ILE LYS LEU MET ASN PRO GLN ARG SER THR VAL TRP TYR".split()
    aa123 = dict(zip(aa1, aa3))
    # aa321 = dict(zip(aa3, aa1))
    chains = cmd.get_chains(obj)
    seq_cat = ''
    for chain in chains:
        seq = cmd.get_fastastr(f'{obj} and chain {chain}')
        seq = seq.split()[1:]
        seq = ''.join(seq)
        seq_cat += seq
    seq_cat = np.asarray([aa123[r] for r in seq_cat])
    return seq_cat


def get_chain_seq(obj):
    myspace = {'chains': []}
    cmd.iterate(obj, 'chains.append(chain)', space=myspace)
    chains = np.asarray(myspace['chains'])
    return chains


def get_coords(pdbfilename, object, device, selection=None):
    if selection is None:
        selection = f'{object} and name CA'
    else:
        selection = f'{object} and name CA and {selection}'
    try:
        cmd.load(pdbfilename, object=object)
    except pymol.CmdException:
        cmd.fetch(code=pdbfilename, name=object)
    cmd.remove(f'not ({selection}) and {object}')
    coords = cmd.get_coords(selection=object)
    coords = torch.from_numpy(coords)
    coords = coords.to(device)
    return coords


def write_pdb(obj, coords, outfilename, seq=None, resids=None, chains=None):
    cmd.load_coords(coords, obj)
    if seq is not None:
        myspace = {}
        myspace['seq_iter'] = iter(seq)
        cmd.alter(obj, 'resn=f"{seq_iter.__next__()}"', space=myspace)
    if resids is not None:
        myspace = {}
        myspace['resid_iter'] = iter(resids)
        cmd.alter(obj, 'resi=f"{resid_iter.__next__()}"', space=myspace)
    if chains is not None:
        myspace = {}
        myspace['chain_iter'] = iter(chains)
        cmd.alter(obj, 'chain=f"{chain_iter.__next__()}"', space=myspace)
    cmd.save(outfilename, selection=obj)


if __name__ == '__main__':
    import optimap
    import doctest
    import os
    import argparse

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    parser = argparse.ArgumentParser(description='Iterative Closest Point algorithm for structural alignment')
    parser.add_argument(
        '--pdb1',
        type=str,
        help='First protein structure (mobile). If a PDB code is given the file is downloaded from the PDB')
    parser.add_argument(
        '--pdb2',
        type=str,
        help='Second protein structure (reference). If a PDB code is given the file is downloaded from the PDB')
    parser.add_argument('--sel1', default='all')
    parser.add_argument('--sel2', default='all')
    parser.add_argument(
        '--pdbs',
        type=str,
        nargs='+',
        help='Pairwise RMSD calculation using ICP. If a PDB code is given the file is downloaded from the PDB')
    parser.add_argument('--niter', type=int, help='Number of iterations (default: 100)', default=100)
    parser.add_argument('--thr', type=float, help='Distance threshold for ICP', default=3.8)
    parser.add_argument(
        '--flex',
        type=float,
        help='Distance threshold for flexible fitting using least square (default=0, no least square flexible fitting)',
        default=0.)
    parser.add_argument('--permute',
                        default=False,
                        help='Permute the coordinates of pdb1 to fit pdb2',
                        action='store_true')
    parser.add_argument('--debug',
                        default=False,
                        help='Just run the doctest for debug purpose only',
                        action='store_true')
    args = parser.parse_args()

    if args.debug:
        print("Debugging...")
        doctest.testmod()
        sys.exit()

    if args.pdbs is not None:
        cmd.set('fetch_path', os.path.expanduser('~/pdb'))
        for i, pdb1 in enumerate(args.pdbs):
            coords_ref = get_coords(pdb1, 'ref', device)
            for pdb2 in args.pdbs[i + 1:]:
                coords_in = get_coords(pdb2, 'mod', device=device)
                cmd.delete('mod')
                coords_out, rmsd = icp(coords_in,
                                       coords_ref,
                                       device,
                                       args.niter,
                                       lstsq_fit_thr=args.flex,
                                       dist_thr=args.thr,
                                       verbose=False)
                print(f"pdb1: {pdb1}")
                print(f"pdb2: {pdb2}")
                print(f"rmsd: {rmsd:.3f}")
                print()
            cmd.delete('ref')
        sys.exit(0)

    if args.pdb1 is None or args.pdb2 is None:
        print("")
        print("The following arguments are required: --pdb1, --pdb2")
        print("")
        parser.print_help()
        sys.exit(1)

    coords_ref = get_coords(args.pdb2, 'ref', device=device, selection=args.sel2)
    coords_in = get_coords(args.pdb1, 'mod', device, selection=args.sel1)
    # Try to align
    # R, t = find_rigid_alignment(coords_in, coords_ref)
    # coords_out = transform(coords_in, R, t)
    # rmsd = get_RMSD(coords_out, coords_ref)
    # print(f'RMSD for rigid alignment: {rmsd}')
    # coords_out = coords_out.cpu().detach().numpy()
    # cmd.load_coords(coords_out, 'mod')
    # cmd.save('out_align.pdb', selection='mod')
    # Try the ICP
    coords_out, rmsd = icp(coords_in, coords_ref, device, args.niter, lstsq_fit_thr=args.flex, dist_thr=args.thr)
    resids_ref = None
    chains_ref = None
    seq_ref = None
    if args.permute:
        _, sel, P = assign_anchors(coords_out, coords_ref, return_perm=True, dist_thr=3.8)
        coords_out = coords_out.T.mm(P).T
        torm = cmd.select('mod') - coords_out.shape[0]
        resids = get_resids('mod')
        chains = get_chain_seq('mod')
        for i in range(torm):
            cmd.remove(f'mod and resi {resids[i]} and chain {chains[i]}')
        resids_ref = get_resids('ref')[sel]
        chains_ref = get_chain_seq('ref')[sel]
        seq_ref = get_sequence('ref')[sel]
    coords_out = coords_out.cpu().detach().numpy()
    write_pdb(obj='mod',
              coords=coords_out,
              outfilename=f'{os.path.splitext(args.pdb1)[0]}_icp.pdb',
              resids=resids_ref,
              seq=seq_ref,
              chains=chains_ref)
