#!/usr/bin/env python3

import gzip

import numpy as np
import scipy.spatial.distance as scidist

x = np.concatenate((np.random.normal(loc=0, scale=2, size=(50, 64)),
                    np.random.normal(loc=5, scale=1, size=(50, 64)),
                    np.random.normal(loc=-2, scale=0.5, size=(50, 64))))
classes = np.concatenate(([0]*50, [1]*50, [2]*50))
print(f"{x.shape=}")
dmat = scidist.squareform(scidist.pdist(x))
print(f"{dmat.shape=}")
npts = x.shape[0]

with gzip.open('data/mds_inp.rec.gz', 'wt') as gz:
    for i in range(npts):
        for j in range(i, npts):
            gz.write(f'{i=}\n')
            gz.write(f'{j=}\n')
            class_i = classes[i]
            class_j = classes[j]
            gz.write(f'{class_i=}\n')
            gz.write(f'{class_j=}\n')
            distance = dmat[i, j]
            gz.write(f'{distance=}\n')
            gz.write('--\n')

with gzip.open('data/mds_pts.rec.gz', 'wt') as gz:
    for i in range(npts):
        class_i = classes[i]
        gz.write(f'{class_i=}\n')
        pt = list(x[i])
        gz.write(f'{pt=}\n')
        gz.write('--\n')
