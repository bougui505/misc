#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Sep 11 10:13:50 2024

import os
import sys
import traceback
from types import FrameType

import numpy as np
import torch


# trace function to print every frame event, the function name, the line and the 
# raw code of that line and local the local variables
class Debugger():
    def __init__(self):
        """"""
        self.printed_var = dict()
    def start(self):
        sys.settrace(self.trace_func_local_vars)
    def trace_func_local_vars(self, frame: FrameType, event, arg, verbose=False):
        """
        See: https://toptechtips.github.io/2023-04-13-trace-local-variable-python-function/
        """
        stack = traceback.extract_stack(limit=2)
        code = traceback.format_list(stack)[0].split('\n')[1].strip()  # gets the source code of the line
        _locals = frame.f_locals
        # print("Event: {0}  Func: {1}, Line: {2}, raw_code: {3}, local_vars: {4}".format(event, 
        #                                                 frame.f_code.co_name,
        #                                                 frame.f_lineno,
        #                                                 code, _locals))
        funcname = frame.f_code.co_name
        lineno = frame.f_lineno
        local_vars = _locals
        filename = frame.f_code.co_filename
        # print(f"{filename}:{funcname}:{lineno}:")
        if verbose:
            print(f"{filename=}")
            print(f"{funcname=}")
            print(f"{lineno=}")
        for k in local_vars:
            v = local_vars[k]
            if k in self.printed_var:
                if v == self.printed_var[k]:
                    continue
            if torch.is_tensor(v):
                print(f"{k}={v.shape}")
            elif isinstance(v, np.ndarray):
                print(f"{k}={v.shape}")
            else:
                print(f"{k}={v}")
            self.printed_var[k] = v
        if verbose:
            print("--")
        return self.trace_func_local_vars
    def stop(self):
        sys.settrace(None)


if __name__ == "__main__":
    import argparse

    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description="")
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument("-a", "--arg1")
    parser.add_argument("--test", help="Test the code", action="store_true")
    parser.add_argument("--func", help="Test only the given function(s)", nargs="+")
    args = parser.parse_args()

    if args.test:
        def do_multiply(a, b):
            return a * b

        def do_add(a, b):
            c = a + b
            return do_multiply(a, c)

        debugger = Debugger()
        debugger.start()
        do_add(1,3)
        debugger.stop()
