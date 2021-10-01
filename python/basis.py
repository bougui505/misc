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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import copy


class Basis():
    """
    Change basis for coordinates

    Attributes:
        origin
        origin_new
        u
        v
        w
        A: transition matrix from new to old basis
        A_inv: transition matrix from old to new basis
        dim: dimension of the space
        coords: coordinates in the basis of origin
        coords_new: coordinates in the new basis
        spherical_coords: spherical coordinates in the new basis

    """
    def __init__(self, u=None, v=None, w=None, origin=np.zeros(3)):
        """

        Args:
            u: x axis of new basis in the old basis
            v: y axis of new basis in the old basis
            w: z axis of new basis in the old basis
            origin: origin of the new basis in the old basis coordinates

        >>> u = [1, 0, 0]
        >>> v = [0, 1, 0]
        >>> w = [0, 0, 1]
        >>> basis = Basis(v, w, u, origin=(3, 4, 5))
        >>> coords = np.arange(15).reshape((5, 3))
        >>> coords
        array([[ 0,  1,  2],
               [ 3,  4,  5],
               [ 6,  7,  8],
               [ 9, 10, 11],
               [12, 13, 14]])
        >>> coords_new = basis.change(coords)
        >>> coords_new
        array([[-3., -3., -3.],
               [ 0.,  0.,  0.],
               [ 3.,  3.,  3.],
               [ 6.,  6.,  6.],
               [ 9.,  9.,  9.]])
        >>> coords_back = basis.back(coords_new)
        >>> (coords_back == coords).all()
        True
        >>> # Optional plot
        >>> # basis.plot()
        >>> # Build a basis from 3 points in 3D
        >>> basis = Basis()
        >>> coords = np.asarray([[1.504, 3.440, 5.674], [0.874, 7.070, 6.635], [4.095, 8.990, 7.462]])
        >>> basis.build(coords)
        >>> coords_new = basis.change(coords)
        >>> coords_new
        array([[-5.33348584, -3.50237831,  0.        ],
               [-3.83994401,  0.        ,  0.        ],
               [ 0.        ,  0.        ,  0.        ]])
        >>> # Get spherical coordinates in the new basis
        >>> spherical_coords = basis.spherical
        >>> spherical_coords
        array([[ 6.3806524 ,  1.57079633,  0.58105487],
               [ 3.83994401,  1.57079633, -0.        ],
               [ 0.        ,  1.57079633,  0.        ]])
        >>> coords_back = basis.back(coords_new)
        >>> np.allclose(coords_back, coords)
        True
        >>> # Optional plot
        >>> # basis.plot()
        >>> # Try for a new point outside of the plan
        >>> coords = [5.217, 9.211, 11.085]
        >>> coords_new = basis.change(coords)
        >>> coords_new
        array([[ 1.83192853,  0.24012572, -3.3196734 ]])
        >>> coords_back = basis.back(coords_new)
        >>> np.allclose(coords_back, coords)
        True
        >>> basis.spherical
        array([[3.79919123, 2.63372654, 0.13033504]])
        >>> basis.set_spherical(basis.spherical)
        >>> # Test if back calculated cartesian coordinates from spherical in new basis match the original
        >>> np.allclose(basis.coords_new, coords_new)
        True
        >>> np.allclose(basis.coords, coords)
        True

        """
        self.u, self.v, self.w = u, v, w
        self.origin = origin
        if self.u is not None and self.v is not None and self.w is not None:
            self._set()

    def _set(self):
        self.dim = len(self.u)
        self.A = np.c_[self.u, self.v, self.w]  # transition matrix
        assert np.allclose(
            self.A.T.dot(self.A),
            np.identity(self.dim),
            rtol=1e-04,
            atol=1e-07
        ), f"(u, v, w) is not an orthonormal basis: {self.A.T.dot(self.A)}"
        self.A_inv = np.linalg.inv(self.A)
        self.origin = np.asarray(self.origin)[None, ...]
        self.origin_new = self.A_inv.dot(self.origin.T).T
        self.coords = None  # Coords in the first basis
        self.coords_new = None  # Coords in the new basis
        self.spherical_coords = None  # Spherical coordinates in the new basis

    def _set_coords(self, coords):
        out = np.asarray(coords).copy()
        if out.ndim == 1:
            out = out[None, ...]
        return out

    @property
    def spherical(self):
        """
        Get the spherical coordinates in the basis

        """
        x, y, z = self.coords_new.T
        r = np.linalg.norm(self.coords_new, axis=1)
        theta = np.arccos(np.divide(z, r, out=np.zeros_like(r),
                                    where=(r != 0)))
        phi = np.arctan(np.divide(y, x, out=np.zeros_like(y), where=(x != 0)))
        spherical_coords = np.c_[r, theta, phi]
        self.spherical_coords = spherical_coords
        return spherical_coords

    def set_spherical(self, spherical_coords):
        """
        Set the spherical coords in the basis and compute the corresponding
        cartesian coordinates in the basis (self.coords_new) and in the
        original basis (self.coords)

        Args:
            spherical_coords: ndarray with (r, theta, phi)

        """
        self.spherical_coords = self._set_coords(spherical_coords)
        r, theta, phi = self.spherical_coords.T
        assert (theta >= 0).all() and (theta <= np.pi).all()
        assert (phi >= 0).all() and (phi <= 2 * np.pi).all()
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        self.coords_new = self._set_coords(np.c_[x, y, z])
        self.back(self.coords_new)

    def change(self, coords):
        """

        Args:
            coords: Coordinates of points in the old basis (shape: (n, self.dim))

        """
        self.coords = self._set_coords(coords)
        coords_new = self.A_inv.dot(self.coords.T).T - self.origin_new
        self.coords_new = self._set_coords(coords_new)
        return coords_new

    def back(self, coords_new):
        """

        Args:
            coords_new: Coordinates of points in the new basis (shape: (n, self.dim))

        """
        incoords = coords_new.copy()
        self.coords_new = self._set_coords(incoords)
        incoords += self.origin_new
        coords = self.A.dot(incoords.T).T
        self.coords = self._set_coords(coords)
        return coords

    def build(self, coords):
        """Build a basis from 3 points (only in 3D)

        Args:
            coords: np array with shape (3, 3).
                    The first dimension is for the number of points,
                    the second for the dimension of the space.

        """
        assert coords.shape == (3, 3)
        a = coords[1] - coords[0]
        b = coords[2] - coords[1]
        assert np.linalg.norm(np.cross(a, b)) != 0
        self.origin = coords[2]
        u = b.copy()
        u /= np.linalg.norm(u)
        w = np.cross(a, b)
        w /= np.linalg.norm(w)
        v = np.cross(u, w)
        v /= np.linalg.norm(v)
        self.u, self.v, self.w = u, v, w
        self._set()

    def plot(self):
        if self.dim == 3:
            ax = plt.axes(projection='3d')
            ax.scatter3D(self.coords[:, 0], self.coords[:, 1], self.coords[:,
                                                                           2])
            # ax.quiver(self.origin[0, 0], self.origin[0, 1], self.origin[0, 2],
            #           self.u, self.v, self.w)
            for v in [self.u, self.v, self.w]:
                ax.plot3D([self.origin[0, 0], self.origin[0, 0] + v[0]],
                          [self.origin[0, 1], self.origin[0, 1] + v[1]],
                          [self.origin[0, 2], self.origin[0, 2] + v[2]])
            plt.show()


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--test', help='Test the module', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod()
        sys.exit()
