#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#############################################################################

import numpy as np
from pymol import cmd


def create_topology(coords, object):
    for i, c in enumerate(coords):
        cmd.pseudoatom(object,
                       pos=list(c),
                       resi=i + 1,
                       name='CA',
                       resn='GLY',
                       hetatm=False)
        if i > 0:
            cmd.bond(f'resi {i + 1}', f'resi {i}')
    cmd.color('green', object)
    cmd.orient(object)


@cmd.extend
def load_traj_npy(npyfile, topology=None, selection='all', object='traj'):
    """
    load the given npy file containing a numpy array of shape (nstep, natoms, 3)
    if topology is None assess a linear topology
    """
    traj = np.load(npyfile)
    if topology is None:
        coords = traj[0]
        print(coords.shape)
        create_topology(coords, object)
    else:
        cmd.load(filename=topology, object=object)
        cmd.remove(f'{object} and not {selection}')
    startindex = 1
    nsteps = len(traj)
    state = 2
    for i in range(startindex, nsteps):
        coords = traj[i]
        cmd.create(object,
                   selection=object,
                   source_state=state - 1,
                   target_state=state)
        cmd.load_coords(coords, object, state=state)
        state += 1


if __name__ == '__main__':
    import argparse
    import doctest
    import sys

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS
                        | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
