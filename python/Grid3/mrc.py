#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

import mrcfile
import numpy as np
import sys


def save_density(density, outfilename, spacing=1, origin=[0, 0, 0], padding=0):
    """
    Save the density file as mrc for the given atomname
    """
    density = density.astype('float32')
    with mrcfile.new(outfilename, overwrite=True) as mrc:
        mrc.set_data(density.T)
        mrc.voxel_size = spacing
        mrc.header['origin']['x'] = origin[0] - padding + .5 * spacing
        mrc.header['origin']['y'] = origin[1] - padding + .5 * spacing
        mrc.header['origin']['z'] = origin[2] - padding + .5 * spacing
        mrc.update_header_from_data()
        mrc.update_header_stats()


def mrc_to_array(mrcfilename):
    """
    Print the MRC values on stdout
    """
    with mrcfile.open(mrcfilename) as mrc:
        return mrc.data


if __name__ == '__main__':
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--npy', help='Read a npy file and create a MRC file (see: --out)')
    parser.add_argument('--origin', type=float, nargs='+', default=[0, 0, 0], help='Origin for the MRC file')  # 10. 20. 30.
    parser.add_argument('--spacing', type=float, default=1, help='Spacing for the MRC file')  # 1.
    parser.add_argument('--out', help='Out MRC file name (see: --npy)')  # 10. 20. 30.
    parser.add_argument('--mrc', help='Read the given MRC and print the flatten data to stdout')
    args = parser.parse_args()

    if args.npy is not None:
        data = np.load(args.npy)
        save_density(data, args.out, args.spacing, args.origin, 0)
    if args.mrc is not None:
        data = mrc_to_array(args.mrc)
        np.savetxt(sys.stdout, data.flatten())
