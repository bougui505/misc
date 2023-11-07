#!/usr/bin/env python3

import numpy as np
import scipy.spatial.distance as scidist


def getdist(data):
    raw = data['pt']
    X = []
    for l in raw:
        l = l.replace('[', '').replace(']', '')
        l = [float(e) for e in l.split(', ')]
        X.append(l)
    X = np.asarray(X)
    dmat = scidist.squareform(scidist.pdist(X))
    return dmat
