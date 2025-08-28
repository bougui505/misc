import torch
import torch.utils.benchmark as benchmark
from itertools import product

def batched_dot_mul_sum(a, b):
    '''Computes batched dot by multiplying and summing'''
    return a.mul(b).sum(-1)

def batched_dot_bmm(a, b):
    '''Computes batched dot by reducing to ``bmm``'''
    a = a.reshape(-1, 1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1], 1)
    return torch.bmm(a, b).flatten(-3)

# Input for benchmarking to ensure correctness
x = torch.randn(10000, 64)
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
    x_input = torch.ones((b, n))
    for num_threads in [1, 4, 16, torch.get_num_threads()]: # Use actual max threads if available
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
