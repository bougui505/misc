#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import faiss
import numpy as np
import scipy.spatial.distance as scidist

D = 128
N = 100000

N_train = 10000
# Param of PQ
M = 32  # The number of sub-vector. Typically this is 8, 16, 32, etc.
nbits = 4  # bits per sub-vector. This is typically 8, so that each sub-vec is encoded by 1 byte
# Param of IVF
nlist = int(np.sqrt(N))  # The number of cells (space partition). Typical value is sqrt(N)
# Param of HNSW
hnsw_m = 32  # The number of neighbors for HNSW. This is typically 32

topk = 4

N_search = 3  # Number of vectors to search for

# Show params
print("D:", D)
print("N:", N)
print("N_train:", N_train)
print("M:", M)
print("nbits:", nbits)
print("nlist:", nlist)
print("topk:", topk)
print("N_search:", N_search)

X = np.random.random((N, D)).astype(np.float32)

# Setup
# index = faiss.IndexFlatL2(D)
index = faiss.IndexPQFastScan(D, M, nbits)

# Train
index.train(X)

# Add
index.add(X)
# index.add_with_ids(X, ids)

queries = np.random.random((N_search, D)).astype(np.float32)
# queries = X[:N_search]

# Search
dists, ids = index.search(x=queries, k=topk)
print('FAISS results:')
# print(dists)
print(ids)

print('scidist results:')
dmat = scidist.cdist(queries, X)
print(dmat.shape)
ids_2 = dmat.argsort(axis=1)[:, :topk]
dists_2 = dmat[:, ids.flatten()]
# print(dists_2)
print(ids_2)
