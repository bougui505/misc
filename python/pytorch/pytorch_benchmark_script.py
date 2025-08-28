#!/usr/bin/env python3
import torch
import torch.utils.benchmark as benchmark
from itertools import product
import os

# Determine the device to run on (GPU if available, otherwise CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

# Input for benchmarking to ensure correctness
x = torch.randn(10000, 64, device=device)
assert batched_dot_mul_sum(x, x).allclose(batched_dot_bmm(x, x))

# Compare takes a list of measurements which we'll save in results.
results = []

sizes = [1, 64, 1024, 10000]
for b, n in product(sizes, sizes):
    # label and sub_label are the rows
    # description is the column
    label = 'Batched dot'
    sub_label = f'[{b}, {n}]'
    # Use torch.ones for consistent input data during comparison
    x_input = torch.ones((b, n), device=device)
    # When using GPU, num_threads is usually 1 unless specific multi-threading is desired for CPU tasks,
    # but the primary computations will be on the GPU.
    # We still iterate through num_threads for CPU benchmarking cases or if device is 'cpu'.
    if device == "cpu":
        num_threads_list = torch.round(torch.linspace(1, os.cpu_count(), 4)).int()
    else:
        num_threads_list = [1]
    for num_threads in num_threads_list:
        results.append(benchmark.Timer(
            stmt='batched_dot_mul_sum(x_input, x_input)',
            setup='from __main__ import batched_dot_mul_sum',
            globals={'x_input': x_input},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='mul/sum',
        ).blocked_autorange(min_run_time=1))
        results.append(benchmark.Timer(
            stmt='batched_dot_bmm(x_input, x_input)',
            setup='from __main__ import batched_dot_bmm',
            globals={'x_input': x_input},
            num_threads=num_threads,
            label=label,
            sub_label=sub_label,
            description='bmm',
        ).blocked_autorange(min_run_time=1))

compare = benchmark.Compare(results)
compare.trim_significant_figures()
compare.colorize()
compare.print()
