#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2023 Institut Pasteur                                       #
#############################################################################

import gzip
import os

import numpy as np
from numpy import linalg
from sklearn.manifold import TSNE, Isomap


def tsne(data,
         n_components=2,
         perplexity=30.0,
         metric="euclidean",
         init="pca",
         verbose=0):
    if metric == "precomputed":
        init = "random"
    embedder = TSNE(n_components=n_components,
                    perplexity=perplexity,
                    metric=metric,
                    init=init,
                    verbose=verbose)
    out = embedder.fit_transform(data)
    return out


def isomap(data, n_components=2, metric="euclidean"):
    embedder = Isomap(n_components=n_components, metric=metric)
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
        np.savez(outfilename,
                 eigenvalues=eigenvalues,
                 eigenvectors=eigenvectors)
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

def data_from_rec(recfile, selected_fields):
    data = list()
    keys = []
    with gzip.open(recfile, 'rt') as gz:
        i = 0
        for line in gz:
            line = line.strip()
            if line != "--":
                key, val = line.split("=", maxsplit=1)
                if key in selected_fields:
                    val = np.float_(val.replace("[", "").replace("]", "").split(","))
                    data.append(val)
                    keys.append(key)
            else:
                i += 1
    nkeys = len(selected_fields)
    DATA = np.stack(data)
    assert DATA.shape[0] == nkeys * i, f"{DATA.shape[0]=} does not match the number of keys {nkeys}*{i}"
    return DATA, np.asarray(keys)

def data_to_rec(recfile, selected_fields, out, keys):
    assert len(out) == len(keys), f"out has not the same shape {out.shape=} as keys {len(keys)=}"
    keys_unique = np.unique(keys)
    out = {k:iter(out[keys==k]) for k in keys_unique}
    with gzip.open(recfile, 'rt') as gz:
        for line in gz:
            line = line.strip()
            if line != "--":
                print(line)
                key, _ = line.split("=", maxsplit=1)
                if key in selected_fields:
                    print(f"{key}_{args.method}={list(next(out[key]))}")
            else:
                print("--")




if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(
        description="Use various projection methods (see: --method) to project the data in stdin to a low-dimensional space."
    )
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("--rec", help="Read the data from the given rec file. The syntax is --rec recfile field1 [field2] [...]",
                        nargs="+")
    parser.add_argument(
        "--method",
        help="Projection method to use. For TSNE, see: https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html",
        choices=["pca", "tsne", "isomap"],
        default="pca",
    )
    parser.add_argument(
        "--metric",
        help="metric to use for the TSNE (see: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html)",
        type=str,
        default="euclidean",
    )
    parser.add_argument("--dot",
                        help="compute pairwise dot product between data",
                        action="store_true")
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
        "--verbose",
        type=int,
        default=0,
        help="Verbose level for tsne (see sklearn documentation)")
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
    parser.add_argument("--func",
                        help="Test only the given function(s)",
                        nargs="+")
    args = parser.parse_args()

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS
                            | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS
                    | doctest.REPORT_ONLY_FIRST_FAILURE,
                )
        sys.exit()

    if args.rec is None:
        DATA = np.genfromtxt(sys.stdin, dtype=str)
    else:
        recfile = args.rec[0]
        fields = args.rec[1:]
        DATA, KEYS = data_from_rec(recfile=recfile, selected_fields=fields)
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
    print(f"# {DATA.shape=}", file=sys.stderr)
    if args.method == "pca":
        eigenvalues, eigenvectors = compute_pca(DATA)
        OUT = project(DATA, eigenvectors, ncomp=args.n_components)
    if args.method == "tsne":
        OUT = tsne(DATA,
                   n_components=args.n_components,
                   perplexity=args.perplexity,
                   metric=args.metric,
                   verbose=args.verbose)
    if args.method == "isomap":
        OUT = isomap(
            DATA,
            n_components=args.n_components,
            metric=args.metric,
        )
    if args.dot:
        OUT = DATA.dot(DATA.T)
    if args.rec is None:
        print_result(OUT, labels=LABELS, text=TEXT)
    else:
        data_to_rec(recfile=recfile, selected_fields=fields, out=OUT, keys=KEYS)
