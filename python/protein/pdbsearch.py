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
import numpy as np

SEARCHURL = "http://search.rcsb.org/rcsbsearch/v2/query"  # "https://search.rcsb.org/rcsbsearch/v2/query?json="
DATAURL = "https://data.rcsb.org/rest/v1/core/entry"


def __make_request__(params, url):
    response = requests.get(url, {"json": json.dumps(params, separators=(",", ":"))})
    return response


def __print_response__(r):
    pretty_json = json.loads(r.text)
    print(json.dumps(pretty_json, indent=2))


def max_result(n):
    params = {"request_options": {"paginate": {"start": 0, "rows": n}}}
    return params


def structure(entry_id, url=SEARCHURL, operator='relaxed_shape_match', max_results=10, verbose=False):
    """
    Performs fast searches matching a global 3D shape of assemblies or chains of a given entry (identified by PDB ID), in either strict (strict_shape_match) or relaxed (relaxed_shape_match) modes, using a BioZernike descriptor strategy.

    >>> r = structure('1ycr')
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


def textsearch(text, url=SEARCHURL, max_results=10, verbose=False, fields=['title']):
    """
    Performs text search in the pdb

    """
    params = {
        "query": {
            "type": "terminal",
            "service": "full_text",
            "parameters": {
                "value": text
            }
        },
        "request_options": {
            "return_all_hits": True
        },
        "return_type": "entry"
    }
    params.update(max_result(max_results))
    # See: https://github.com/sbliven/rcsbsearch/blob/c7f8cb7e9f26ed5c78af1688af972fd345de8978/rcsbsearch/search.py#L1024
    r = __make_request__(params, url=url)
    if verbose:
        results = r.json()['result_set']
        header = '#id\t#score'
        for field in fields:
            header += f'\t#{field}'
        print(header)
        for d in results:
            outstr = ''
            pdb = d['identifier']
            score = d['score']
            outstr += f'{pdb}\t{score:.2f}'
            if len(fields) > 0:
                data = data_request(pdb)
            if 'title' in fields:
                outstr += f'\t{get_title(data)}'
            if 'ligand' in fields:
                outstr += f'\t{get_ligands(data)}'
            print(outstr)
    return r


def get_title(data):
    """
    Returns the title field from the data structure returned by data_request
    """
    return data['struct']['title']


def get_ligands(data):
    try:
        liglist = data['rcsb_binding_affinity']
        liglist = np.unique([e['comp_id'] for e in liglist])
    except KeyError:
        liglist = []
    return liglist


def print_data(data, keys=[]):
    if len(keys) == 0:
        # required to be able to parse the output with jq:
        # See: https://github.com/stedolan/jq/issues/501
        print(str(data).replace("'", '"'))
    else:
        outstr = ''
        for key in keys:
            if key == 'title':
                outstr += f'{key}: {get_title(data)}\n'
            if key == 'ligand':
                ligands = get_ligands(data)
                for ligand in ligands:
                    outstr += f'{key}: {ligand}\n'
        print(outstr)


def data_request(pdb, url=DATAURL):
    """
    >>> results = data_request('1t4e')
    >>> results.keys()
    dict_keys(['audit_author', 'cell', 'citation', 'diffrn', 'diffrn_detector', 'diffrn_radiation', 'diffrn_source', 'entry', 'exptl', 'exptl_crystal', 'exptl_crystal_grow', 'pdbx_audit_revision_category', 'pdbx_audit_revision_details', 'pdbx_audit_revision_group', 'pdbx_audit_revision_history', 'pdbx_audit_revision_item', 'pdbx_database_related', 'pdbx_database_status', 'pdbx_vrpt_summary', 'rcsb_accession_info', 'rcsb_entry_container_identifiers', 'rcsb_entry_info', 'rcsb_primary_citation', 'refine', 'refine_analyze', 'refine_hist', 'refine_ls_restr', 'reflns', 'reflns_shell', 'software', 'struct', 'struct_keywords', 'symmetry', 'rcsb_id', 'rcsb_binding_affinity'])
    >>> results['rcsb_entry_info']
    {'assembly_count': 1, 'branched_entity_count': 0, 'cis_peptide_count': 0, 'deposited_atom_count': 1682, 'deposited_hydrogen_atom_count': 0, 'deposited_model_count': 1, 'deposited_modeled_polymer_monomer_count': 192, 'deposited_nonpolymer_entity_instance_count': 2, 'deposited_polymer_entity_instance_count': 2, 'deposited_polymer_monomer_count': 192, 'deposited_solvent_atom_count': 50, 'deposited_unmodeled_polymer_monomer_count': 0, 'diffrn_radiation_wavelength_maximum': 1.5418, 'diffrn_radiation_wavelength_minimum': 1.5418, 'disulfide_bond_count': 0, 'entity_count': 3, 'experimental_method': 'X-ray', 'experimental_method_count': 1, 'inter_mol_covalent_bond_count': 0, 'inter_mol_metalic_bond_count': 0, 'molecular_weight': 23.47, 'na_polymer_entity_types': 'Other', 'nonpolymer_entity_count': 1, 'nonpolymer_molecular_weight_maximum': 0.58, 'nonpolymer_molecular_weight_minimum': 0.58, 'polymer_composition': 'homomeric protein', 'polymer_entity_count': 1, 'polymer_entity_count_dna': 0, 'polymer_entity_count_rna': 0, 'polymer_entity_count_nucleic_acid': 0, 'polymer_entity_count_nucleic_acid_hybrid': 0, 'polymer_entity_count_protein': 1, 'polymer_entity_taxonomy_count': 1, 'polymer_molecular_weight_maximum': 11.16, 'polymer_molecular_weight_minimum': 11.16, 'polymer_monomer_count_maximum': 96, 'polymer_monomer_count_minimum': 96, 'resolution_combined': [2.6], 'selected_polymer_entity_types': 'Protein (only)', 'software_programs_combined': ['CNX', 'PROTEUM PLUS'], 'solvent_entity_count': 1, 'structure_determination_methodology': 'experimental', 'structure_determination_methodology_priority': 10, 'diffrn_resolution_high': {'provenance_source': 'Depositor assigned', 'value': 2.4}}
    >>> results['struct']
    {'pdbx_descriptor': 'Ubiquitin-protein ligase E3 Mdm2 (E.C.6.3.2.-)', 'title': 'Structure of Human MDM2 in complex with a Benzodiazepine Inhibitor'}
    """
    response = requests.get(f'{url}/{pdb}')
    results = response.json()
    return results


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
    parser.add_argument('--data', help='Print all the data available for the given pdb', action='store_true')
    parser.add_argument('--keys',
                        help='List of data keys to print out. For text search, available keys are: - title',
                        nargs='+',
                        default=[])
    parser.add_argument(
        '--structure',
        help=
        'Performs fast searches matching a global 3D shape of assemblies or chains of a given entry (identified by PDB ID), in either strict (strict_shape_match) or relaxed (relaxed_shape_match) modes, using a BioZernike descriptor strategy.',
        action='store_true')
    parser.add_argument('--text', help='Text search in the PDB')
    parser.add_argument('-n',
                        '--max_results',
                        help='maximum number of results to return (default=10)',
                        type=int,
                        default=10)
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    if args.data:
        data = data_request(args.pdb)
        print_data(data, keys=args.keys)
        # print_pdb_data(r, filters=args.keys)
    if args.structure:
        r = structure(args.pdb, operator='relaxed_shape_match', verbose=True, max_results=args.max_results)
    if args.text is not None:
        r = textsearch(args.text, verbose=True, max_results=args.max_results, fields=args.keys)

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f,
                                               globals(),
                                               optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
