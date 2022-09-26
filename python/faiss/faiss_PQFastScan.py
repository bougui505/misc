#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import faiss
import numpy as np
from misc.Timer import Timer
from misc.randomgen import randomstring
import tqdm

timer = Timer(autoreset=True)

D = 128
N = 100000000

N_batch = 100
N_train = 10000
# Param of PQ
M = 16  # The number of sub-vector. Typically this is 8, 16, 32, etc.
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
print("N_batch:", N_batch)
print("N_train:", N_train)
print("M:", M)
print("nbits:", nbits)
print("nlist:", nlist)
print("topk:", topk)
print("N_search:", N_search)

timer.start('Random data generation')
Xt = np.random.random((N_train, D)).astype(np.float32)
# X = np.random.random((N, D)).astype(np.float32)
timer.stop()

# Setup
# quantizer = faiss.IndexHNSWFlat(D, hnsw_m)
# index = faiss.IndexIVFPQ(quantizer, D, nlist, M, nbits)
# index = faiss.IndexFlatL2(D)  # EXACT
index = faiss.IndexPQFastScan(D, M, nbits)

# timer.start('Generating random ids')
# ids = np.asarray([randomstring(5) for i in range(len(X))]).astype(np.str_)
# timer.stop()

# Train
timer.start('Training index')
index.train(Xt)
timer.stop()

# Add
timer.start('Adding data to index')
pbar = tqdm.tqdm(total=N_batch)
queries = []
ind_queries = []
for i in range(N_batch):
    X = np.random.random((N // N_batch, D)).astype(np.float32)
    index.add(X)
    if len(queries) < N_search:
        if np.random.uniform() > 0.5:
            choice = np.random.choice(len(X))
            queries.append(X[choice])
            ind_queries.append(i * len(X) + choice)
    pbar.update(1)
pbar.close()
# index.add_with_ids(X, ids)
timer.stop()
queries = np.asarray(queries)
print(queries.shape)

# Search
timer.start('Searching')
dists, ids = index.search(x=queries, k=topk)
print(ind_queries)
print(dists)
print(ids)
timer.stop()

# Show params
print("D:", index.d)
print("N:", index.ntotal)
# print("M:", index.pq.M)
# print("nbits:", index.pq.nbits)
# print("nlist:", index.nlist)
