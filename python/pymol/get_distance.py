#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#############################################################################

from pymol import cmd
import scipy.spatial.distance as scidist
import matplotlib.pyplot as plt


def get_distance(sel1, sel2):
    nframes = cmd.count_states()
    print('Number of frames in traj', nframes)
    coords1 = cmd.get_coords(selection=sel1, state=0)
    natoms1 = coords1.shape[0] // nframes
    print('Number of atoms in sel1', natoms1)
    coords1 = coords1.reshape((nframes, natoms1, 3))
    coords2 = cmd.get_coords(selection=sel2, state=0)
    natoms2 = coords2.shape[0] // nframes
    print('Number of atoms in sel2', natoms2)
    coords2 = coords2.reshape((nframes, natoms2, 3))
    out = []
    for i in range(nframes):
        dists = scidist.cdist(coords1[i], coords2[i])
        dist = dists.min()
        out.append(dist)
    return out


def plot_dists(dists):
    plt.plot(dists)
    plt.xlabel('Steps')
    plt.ylabel('Distance (Å)')
    plt.show()


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--top', help='Topology file')
    parser.add_argument('--traj', help='Trajectory file')
    parser.add_argument(
        '--sel1',
        help=
        'Selection of first atom set. For help in selection see: https://pymolwiki.org/index.php/Selection_Algebra'
    )
    parser.add_argument('--sel2', help='Selection of second atom set')
    args = parser.parse_args()

    cmd.load(filename=args.top, object='trajectory')
    cmd.load_traj(filename=args.traj,
                  object='trajectory',
                  state=1,
                  selection=f'{args.sel1} or {args.sel2}')

    dists = get_distance(args.sel1, args.sel2)
    plot_dists(dists)
