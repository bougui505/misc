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

from sklearn.decomposition import PCA
import numpy as np
from scipy.interpolate import griddata
import kabsch
# from scipy.stats import binned_statistic_2d


def grid(x, y, z, resX=100, resY=100):
    "Convert 3 column data to matplotlib grid"
    xi = np.linspace(min(x), max(x), resX)
    yi = np.linspace(min(y), max(y), resY)
    X, Y = np.meshgrid(xi, yi)
    Z = griddata(np.c_[x, y], z, (X, Y), method='linear')
    # Z, xi, yi, _ = binned_statistic_2d(x, y, z, bins=(100, 100), statistic='max')
    # X, Y = np.meshgrid(xi[:-1], yi[:-1])
    return X, Y, Z


class View(object):
    """
    Center the frame of the protein for projection
    """
    def __init__(self, dopca=True, center=None):
        """
        If dopca is True, compute PCA to center the view (the default)
        if center is a set of coordinates, align the center of mass of the coordinates with the x-axis
        """
        self.dopca = dopca
        self.center = center
        if self.center is not None:
            self.dopca = False
            self.center = np.mean(center, axis=0)
        self.pca = PCA(n_components=3)

    def fit(self, X):
        if self.dopca:
            self.pca.fit(X)
        if self.center is not None:
            self.mu = np.mean(X, axis=0)
            center_c = self.center - self.mu
            r = np.linalg.norm(center_c)
            mob = np.stack((np.asarray([0, 0, 0]), center_c))
            target = np.asarray([[0, 0, 0], [r, 0, 0]])
            # print(mob.shape, target.shape)
            self.R, self.t = kabsch.rigid_body_fit(mob, target)
            # print(self.R, self.t)

    def transform(self, X):
        if self.dopca:
            return self.pca.transform(X)
        if self.center is not None:
            X_c = X - self.mu
            return (self.R.dot(X_c.T)).T + self.t


class Miller(object):
    """
    see: https://bmcstructbiol.biomedcentral.com/articles/10.1186/s12900-016-0055-7#Sec1
    """
    def __init__(self, center=None, spheric=True):
        self.spheric = spheric
        self.view = View(dopca=True, center=center)
        self.n_circles = 13

    def fit(self, X):
        self.view.fit(X)
        X = self.view.transform(X)
        self.r = np.linalg.norm(X, axis=1).max()

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __proj__(self, X):
        lat = np.arctan(X[:, 2] / self.r)
        lon = np.arctan(X[:, 1] / X[:, 0])
        xp = lon
        yp = (5 / 4) * np.log(np.tan(np.pi / 4 + (2 / 5) * lat))
        xy = np.c_[xp, yp]
        return xy

    def transform(self, X):
        X = self.view.transform(X)
        self.alt = np.linalg.norm(X, axis=1)
        if self.spheric:
            X = self.r * X / self.alt[:, None]
        xy = self.__proj__(X)
        return xy

    def latitude_circles(self):
        lcs = latitude_circles(self.r, self.n_circles, np.zeros(3))
        xy = self.__proj__(np.concatenate(lcs))
        return xy

    def longitude_circles(self):
        lcs = longitude_circles(self.r, self.n_circles, np.zeros(3))
        xy = self.__proj__(np.concatenate(lcs))
        return xy

    def grid(self):
        latc = self.latitude_circles()
        lonc = self.longitude_circles()
        plt.scatter(*latc.T, s=.05, c='gray')
        plt.scatter(*lonc.T, s=.05, c='gray')
        xmax = np.nanmax(latc, axis=0)[0]
        sel = (latc[:, 0] == xmax)
        xmax = latc[:, 0][sel]
        ymax = latc[:, 1][sel]
        xticklabel = -self.r / 2 + 0.5 * self.r / self.n_circles
        for i, (xl, yl) in enumerate(zip(xmax, ymax)):
            xticklabel += self.r / self.n_circles
            plt.annotate(f'{xticklabel:.2f}', (xl, yl), size=10., va='center',
                         xytext=(xl + 0.05, yl), color='gray')
            if i == self.n_circles // 2 - 1:
                plt.annotate(f'latitude (Å)', (xl, yl), size=10.,
                             xytext=(xl + 0.4, yl), rotation='vertical',
                             rotation_mode=None, ha='center', va='center',
                             color='gray')


def circle(p, r, v1, v2, npts=1000, amin=0., amax=2 * np.pi):
    """
    see: https://math.stackexchange.com/a/1184089/192193
    - p: point in 3D
    - v1, v2: orthogonal unit vectors
    - r: scalar
    circle with center p and radius r lying in the plane parallel to v1 and v2
    """
    t = np.linspace(amin, amax, num=npts)[:, None]
    pts = p[None, :] + r * np.cos(t) * v1[None, :] + r * np.sin(t) * v2[None, :]
    return pts


def latitude_circles(R, num, c):
    zs = np.linspace(-R, R, num=num)
    latcirc = []
    for z in zs:
        r = np.sqrt(R**2 - z**2)
        p = c + np.asarray([0, 0, z])
        v1 = np.asarray([1, 0, 0])
        v2 = np.asarray([0, 1, 0])
        latcirc.append(circle(p, r, v1, v2))
    return latcirc


def longitude_circles(R, num, c):
    ys = np.linspace(-R, R, num=num)
    loncirc = []
    for y in ys:
        r = np.sqrt(R**2 - y**2)
        p = c + np.asarray([0, y, 0])
        v1 = np.asarray([1, 0, 0])
        v2 = np.asarray([0, 0, 1])
        da = 0.9 * 2 * np.pi / 3
        loncirc.append(circle(p, r, v1, v2, amin=da, amax=2 * np.pi - da))
    return loncirc


if __name__ == '__main__':
    import pdbsurf
    from pymol import cmd
    import matplotlib.pyplot as plt
    import misc.recutils as recutils
    import misc.Clustering
    from termcolor import colored
    import glob
    import argparse
    import sys
    cmd.feedback('disable', 'all', 'everything')
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-p', '--pdb')
    parser.add_argument('-s', '--sel')
    parser.add_argument('-c', '--caption')
    parser.add_argument('--size', type=float, help='Size of the dots for the scatter plot (default=1.)', default=1.)
    parser.add_argument('--levels', type=int, help='Number of levels in the contour plot (default=10)', default=10)
    parser.add_argument('--center', help='Center the projection on the given selection')
    parser.add_argument('--atomic', help='Project atomic coordinates for selection in caption instead of surface', action='store_true')
    parser.add_argument('--spheric', help='Project the protein surface on a sphere', action='store_true')
    parser.add_argument('--geom', help='Project the geometric center of the caption selections', action='store_true')
    parser.add_argument('--save', help='Save as a figure')
    args = parser.parse_args()

    if args.center is not None:
        cmd.load(args.pdb)
        args.center = cmd.get_coords(args.center)
        cmd.reinitialize()
    miller = Miller(center=args.center, spheric=args.spheric)
    surfpts = pdbsurf.pdb_to_surf(args.pdb, args.sel)
    proj = miller.fit_transform(surfpts)
    X, Y, Z = grid(proj[:, 0], proj[:, 1], miller.alt)
    plt.contourf(X, Y, Z, cmap='coolwarm', levels=args.levels)
    clb = plt.colorbar()
    clb.set_label('z (Å)')
    if args.caption is not None:
        captions = recutils.load(args.caption)
        print("    ---------------------------------------------------------------")
        for caption in captions:
            sel = caption['sel']
            if 'color' in caption:
                color = caption['color']
            else:
                color = None
            if 'pdb' in caption:
                pdbfilename = caption['pdb']
            else:
                pdbfilename = args.pdb
            if 'traj' in caption:
                trajfilename = caption['traj']
            else:
                trajfilename = None
            try:
                print(colored(f'    • {pdbfilename} and {sel}', color))
            except KeyError:
                print(f'    • {pdbfilename} and {sel} -> {color}')
            pdbfilenames = glob.glob(pdbfilename)
            for pdbfilename in pdbfilenames:
                cmd.load(pdbfilename, '_inp_')
            if trajfilename is not None:
                cmd.load_traj(trajfilename, '_inp_', state=1)
            nstates = cmd.count_states('_inp_')
            if 'project' in caption:  # Projection on scattered dots
                project = np.genfromtxt(caption['project'])
                args.atomic = True
                args.geom = True
                assert len(project) == nstates
            else:
                project = None
            if 'label' in caption:  # label of the colorbar for the data projected
                clb_proj_label = caption['label']
            else:
                clb_proj_label = None
            if 'sort' in caption:  # Sort the data based on the 'project' field
                dosort = caption['sort']  # 1: sort | -1: reverse sort
            else:
                dosort = None
            if 'first' in caption:  # Plot only the n-first states
                first = caption['first']
            else:
                first = None
            if 'size' in caption:  # size of the scatter dots
                args.size = caption['size']
            if 'alpha' in caption:  # transparency of the scatter dots
                alpha = caption['alpha']
            else:
                alpha = 1.
            if 'clusters' in caption:
                n_clusters = caption['clusters']
            else:
                n_clusters = None
            if nstates > 1:
                args.atomic = True
            if args.atomic:
                toproj_list = []
                for i in range(nstates):
                    toproj_list.append(cmd.get_coords(f'_inp_ and {sel}', state=i + 1))
                cmd.reinitialize()
            else:
                # TODO: not yet implemented for multistate pdb
                toproj_list = [pdbsurf.pdb_to_surf(pdbfilename, sel), ]
            if args.geom:
                toproj_list = [e.mean(axis=0)[None, :] for e in toproj_list]
            xyz = []
            for i, toproj in enumerate(toproj_list):
                # TODO: remove the loop: project in 1-shot
                sys.stdout.write(f'{i+1}/{len(toproj_list)}\r')
                sys.stdout.flush()
                proj_ = miller.transform(toproj)
                if project is None:
                    xyz.append([proj_[:, 0], proj_[:, 1]])
                else:
                    xyz.append([proj_[:, 0], proj_[:, 1], project[i]])
            print()
            xyz = np.asarray(xyz)
            if n_clusters is not None:  # Kmeans clustering
                labels = misc.Clustering.Kmeans(xyz, n_clusters=n_clusters)
                cluster_inf = ''
                if xyz.shape[1] == 3:
                    z = xyz[:, 2]
                    for label in set(labels):
                        label_sel = (labels == label)
                        if clb_proj_label is not None:
                            unit = clb_proj_label
                        else:
                            unit = 'value'
                        val_mean = z[label_sel].mean()
                        val_std = z[label_sel].std()
                        cluster_inf += f'Population for cluster {label}: {label_sel.sum()}\n'
                        cluster_inf += f'Mean {unit} for cluster {label}: {val_mean:.4g} +- {val_std:.4g}\n'
                    print(cluster_inf)
                project = True
                xyz = np.concatenate((xyz[:, :2], labels[:, None]), axis=1)
                clb_proj_label = "Clusters"
                np.savetxt('miller_clusters.txt', labels, fmt="%d", header=cluster_inf)
            if project is None:
                plt.scatter(xyz[:, 0], xyz[:, 1], s=args.size, color=color, alpha=alpha)
            else:
                if dosort is not None:
                    sorter = np.argsort(xyz[:, 2])
                    if dosort == -1:
                        sorter = sorter[::-1]
                    xyz = xyz[sorter]
                    if first is not None:
                        xyz = xyz[:first]
                if color is None:
                    plt.scatter(xyz[:, 0], xyz[:, 1], s=args.size, c=xyz[:, 2], alpha=alpha)
                    clb_proj = plt.colorbar()
                    if clb_proj_label is not None:
                        clb_proj.set_label(clb_proj_label)
                else:
                    plt.scatter(xyz[:, 0], xyz[:, 1], s=args.size, color=color, alpha=alpha)
        print("    ---------------------------------------------------------------")
    miller.grid()
    plt.axis('off')
    if args.save is None:
        plt.show()
    else:
        plt.savefig(args.save)
