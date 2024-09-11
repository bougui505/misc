#!/usr/bin/env python3

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Sep 11 08:56:45 2024

import os

import torch


def log(msg):
    try:
        logging.info(msg)  # type: ignore
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir

class debug_context():
    """
    Debug context to trace any function calls inside the context
    See: https://stackoverflow.com/a/32261446/1679629
    """

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        print('Entering Debug Decorated func')
        # Set the trace function to the trace_calls function
        # So all events are now traced
        sys.settrace(self.trace_calls)

    def __exit__(self, *args, **kwargs):
        # Stop tracing all events
        sys.settrace = None

    def trace_calls(self, frame, event, arg): 
        # We want to only trace our call to the decorated function
        if event != 'call':
            return
        elif frame.f_code.co_name != self.name:
            return
        # return the trace function to use when you go into that 
        # function call
        return self.trace_lines

    def varprinter(self, local_vars):
        for k in local_vars:
            v = local_vars[k]
            if torch.is_tensor(v):
                print(k, v.shape)
            elif isinstance(v, np.ndarray):
                print(k, v.shape)
            else:
                print(k, v)

    def trace_lines(self, frame, event, arg):
        # If you want to print local variables each line
        # keep the check for the event 'line'
        # If you want to print local variables only on return
        # check only for the 'return' event
        if event not in ['line', 'return']:
            return
        co = frame.f_code
        func_name = co.co_name
        line_no = frame.f_lineno
        filename = co.co_filename
        local_vars = frame.f_locals
        if len(local_vars) > 0:
            print(f"{func_name}:{event} {line_no}")
            self.varprinter(local_vars)
        # print ('  {0} {1} {2} locals: {3}'.format(func_name, 
        #                                           event,
        #                                           line_no, 
        #                                           local_vars))

def debug_decorator(func):
    """ Debug decorator to call the function within the debug context """
    def decorated_func(*args, **kwargs):
        with debug_context(func.__name__):
            return_value = func(*args, **kwargs)
        return return_value
    return decorated_func

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
        import numpy as np

        @debug_decorator
        def testing() : 
            a = 10
            b = 20
            c = a + b
            A = np.zeros((3, 3))

        testing()

        # if args.func is None:
        #     doctest.testmod(
        #         optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE
        #     )
        # else:
        #     for f in args.func:
        #         print(f"Testing {f}")
        #         f = getattr(sys.modules[__name__], f)
        #         doctest.run_docstring_examples(
        #             f,
        #             globals(),
        #             optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE,
        #         )
        sys.exit()
