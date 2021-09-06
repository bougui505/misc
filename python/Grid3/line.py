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


import numpy as np


def get(p0, p1, spacing=1):
    """
    Get the line between p0 and p1
    """
    dt = spacing / np.linalg.norm(p1 - p0)
    tvals = np.arange(0., 1. + dt, dt)
    p0 = p0[..., None]
    p1 = p1[..., None]
    pts = (p1 - p0) * tvals + p0
    pts = np.int_(pts.T)
    if (pts[-1] != p1).any():
        pts = np.r_[pts, np.squeeze(p1)[None, :]]
    return pts


def interpolate(pts, spacing=1):
    """
    Interpolate lines between points (pts)
    """
    lines = []
    for (p0, p1) in zip(pts, pts[1:]):
        lines.extend(get(p0, p1, spacing=spacing))
    lines = np.asarray(lines)
    return lines


if __name__ == '__main__':
    import mrc
    G = np.zeros((10, 10, 10))
    p0 = np.asarray([2, 3, 4])
    p1 = np.asarray([5, 6, 5])
    p2 = np.asarray([8, 7, 6])
    pts = np.c_[p0, p1, p2]
    print('imput:')
    print(pts)
    print('______')
    lines = interpolate(pts)
    print(lines)
    G[tuple(lines.T)] = 1.
    mrc.save_density(G, 'line.mrc')
