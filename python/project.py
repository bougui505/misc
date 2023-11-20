#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#                               				                            #
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
import os

import numpy as np
from numpy import linalg
from sklearn.manifold import TSNE


def tsne(data, n_components=2, perplexity=30.0, metric="euclidean"):
    embedder = TSNE(n_components=n_components, perplexity=perplexity, metric=metric)
    out = embedder.fit_transform(data)
    return out


def compute_pca(X, outfilename=None):
    """
    >>> X = np.random.normal(size=(10, 512))
    >>> proj = compute_pca(X)
    >>> proj.shape
    (10, 2)
    """
    center = X.mean(axis=0)
    cov = (X - center).T.dot(X - center)
    eigenvalues, eigenvectors = linalg.eigh(cov)
    sorter = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[sorter], eigenvectors[:, sorter]
    if outfilename is not None:
        np.savez(outfilename, eigenvalues=eigenvalues, eigenvectors=eigenvectors)
    return eigenvalues, eigenvectors


def project(X, eigenvectors, ncomp=2):
    projection = X.dot(eigenvectors[:, :ncomp])
    return projection


def print_result(out, labels=None, text=None):
    np.savetxt(".tmp", out, fmt="%.4g")
    out = np.loadtxt(".tmp", dtype=str)
    if labels is not None:
        out = np.c_[out, labels]
    if text is not None:
        out = np.c_[out, text]
    np.savetxt(sys.stdout, out, fmt="%s")
    os.remove(".tmp")


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(
        description="Use various projection methods (see: --method) to project the data in stdin to a low-dimensional space."
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument(
        "--method",
        help="Projection method to use. For TSNE, see: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
        choices=["pca", "tsne"],
        default="pca",
    )
    parser.add_argument(
        "--dot", help="compute pairwise dot product between data", action="store_true"
    )
    parser.add_argument(
        "-n",
        "--n_components",
        help="(default: 2) Dimension of the embedded space.",
        default=2,
        type=int,
    )
    parser.add_argument(
        "-p",
        "--perplexity",
        help="(default: 30.0) The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significantly different results. The perplexity must be less than the number of samples.",
        default=30.0,
        type=float,
    )
    parser.add_argument(
        "-t",
        "--text",
        help="Read the last column as text (default: no text). The text is outputed in the last column of the output.",
        action="store_true",
    )
    parser.add_argument(
        "-l",
        "--labels",
        help="Read the last column as labels (default: no labels). The labels are outputed in the last column of the output.",
        action="store_true",
    )
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()

    DATA = np.genfromtxt(sys.stdin, dtype=str)
    if args.text:
        TEXT = DATA[:, -1]
        DATA = DATA[:, :-1]
    else:
        TEXT = None
    if args.labels:
        LABELS = DATA[:, -1]
        DATA = DATA[:, :-1]
    else:
        LABELS = None
    DATA = DATA.astype(float)
    print(f"# {DATA.shape=}")
    if args.method == "pca":
        eigenvalues, eigenvectors = compute_pca(DATA)
        OUT = project(DATA, eigenvectors, ncomp=args.n_components)
    if args.method == "tsne":
        OUT = tsne(DATA, n_components=args.n_components, perplexity=args.perplexity)
    if args.dot:
        OUT = DATA.dot(DATA.T)
    print_result(OUT, labels=LABELS, text=TEXT)
