#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Mar  5 14:51:32 2025

import os

import torch


def truncated_normal(mean, std, low, high, size, uniform_factor=100):
    """
    uniform_factor: Mutliplication factors for the number of points uniformally sampled
    regarding the final number of points sampled

    >>> num_sample = 2000
    >>> samples = truncated_normal(mean=5.0, std=2.0, low=0.0, high=10.0, size=(num_sample,))
    >>> samples.shape
    torch.Size([2000])

    # Sampling num_sample points for 3 different std values
    >>> std = torch.tensor([1.0, 2.0, 3.0]).view((3, 1))
    >>> samples = truncated_normal(mean=5.0, std=std, low=0.0, high=10.0, size=(num_sample,))
    >>> samples.shape
    torch.Size([2000, 3])

    >>> num_distributions = 64
    >>> std = torch.tensor([3.0, 2.0, 1.0]).view(1, 3, 1).expand((num_distributions, 3, 1))
    >>> std.shape
    torch.Size([64, 3, 1])
    >>> samples = truncated_normal(mean=5.0, std=std, low=0.0, high=10.0, size=(num_sample,))
    >>> samples.shape
    torch.Size([2000, 64, 3])

    >>> import matplotlib.pyplot as plt
    >>> samples = samples.swapaxes(0, 1)
    >>> samples.shape
    torch.Size([64, 2000, 3])
    >>> for i, stdv in enumerate(std[0, :, 0]):
    ...     h = plt.hist(samples[:, :, i].flatten(), label=stdv, alpha=0.75, bins=50)
    >>> os.makedirs("figures", exist_ok=True)
    >>> l = plt.legend()
    >>> plt.savefig("figures/truncated_normal.png")
    """
    x_uniform = torch.distributions.uniform.Uniform(low, high).sample((s*uniform_factor for s in size))
    normal = torch.distributions.normal.Normal(loc=mean, scale=std)
    log_p_x = normal.log_prob(x_uniform)
    categorical = torch.distributions.categorical.Categorical(logits=log_p_x)
    return x_uniform[categorical.sample(size)]


def log(msg):
    try:
        logging.info(msg)  # type: ignore
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


if __name__ == "__main__":
    import argparse
    import doctest
    import sys

    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # if not os.path.isdir('logs'):
    #     os.mkdir('logs')
    # logfilename = 'logs/' + os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-a", "--arg1")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f"# {k}: {v}")

    if args.test:
        if args.func is None:
            doctest.testmod(
                optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF
            )
        else:
            for f in args.func:
                print(f"Testing {f}")
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(
                    f,
                    globals(),
                    optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE | doctest.REPORT_NDIFF,
                )
        sys.exit()
