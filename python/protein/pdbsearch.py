#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2022 Institut Pasteur                                       #
#                 				                            #
#                                                                           #
#  Redistribution and use in source and binary forms, with or without       #
#  modification, are permitted provided that the following conditions       #
#  are met:                                                                 #
#                                                                           #
#  1. Redistributions of source code must retain the above copyright        #
#  notice, this list of conditions and the following disclaimer.            #
#  2. Redistributions in binary form must reproduce the above copyright     #
#  notice, this list of conditions and the following disclaimer in the      #
#  documentation and/or other materials provided with the distribution.     #
#  3. Neither the name of the copyright holder nor the names of its         #
#  contributors may be used to endorse or promote products derived from     #
#  this software without specific prior written permission.                 #
#                                                                           #
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS      #
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT        #
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR    #
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT     #
#  HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,   #
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT         #
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,    #
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY    #
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT      #
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE    #
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.     #
#                                                                           #
#  This program is free software: you can redistribute it and/or modify     #
#                                                                           #
#############################################################################

# See: https://search.rcsb.org/#search-api

import requests
import json

URL = "http://search.rcsb.org/rcsbsearch/v2/query"  # "https://search.rcsb.org/rcsbsearch/v2/query?json="


def __make_request__(params, url):
    response = requests.get(url, {"json": json.dumps(params, separators=(",", ":"))})
    return response


def __print_response__(r):
    pretty_json = json.loads(r.text)
    print(json.dumps(pretty_json, indent=2))


def max_result(n):
    params = {"request_options": {"paginate": {"start": 0, "rows": n}}}
    return params


def structure(entry_id, url=URL, operator='relaxed_shape_match', max_results=10, verbose=False):
    """
    Performs fast searches matching a global 3D shape of assemblies or chains of a given entry (identified by PDB ID), in either strict (strict_shape_match) or relaxed (relaxed_shape_match) modes, using a BioZernike descriptor strategy.

    >>> structure('1ycr')
    """
    params = {
        "query": {
            "type": "terminal",
            "service": "structure",
            "parameters": {
                "value": {
                    "entry_id": entry_id,
                    "assembly_id": "1"
                },
                "operator": operator
            }
        },
        "return_type": "entry"
    }
    params.update(max_result(max_results))
    # See: https://github.com/sbliven/rcsbsearch/blob/c7f8cb7e9f26ed5c78af1688af972fd345de8978/rcsbsearch/search.py#L1024
    r = __make_request__(params, url=url)
    if verbose:
        results = r.json()['result_set']
        print('id\tscore')
        for d in results:
            print(f"{d['identifier']}\t{d['score']:.4f}")
    return r


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # ### UNCOMMENT FOR LOGGING ####
    # import os
    # import logging
    # logfilename = os.path.splitext(os.path.basename(__file__))[0] + '.log'
    # logging.basicConfig(filename=logfilename, level=logging.INFO, format='%(asctime)s: %(message)s')
    # logging.info(f"################ Starting {__file__} ################")
    # ### ##################### ####
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('--pdb', help='pdb code to search')
    parser.add_argument(
        '--structure',
        help=
        'Performs fast searches matching a global 3D shape of assemblies or chains of a given entry (identified by PDB ID), in either strict (strict_shape_match) or relaxed (relaxed_shape_match) modes, using a BioZernike descriptor strategy.',
        action='store_true')
    parser.add_argument('-n',
                        '--max_results',
                        help='maximum number of results to return (default=10)',
                        type=int,
                        default=10)
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.structure:
        r = structure(args.pdb, operator='relaxed_shape_match', verbose=True, max_results=args.max_results)

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()