#!/usr/bin/env python3
# -*- coding: UTF8 -*-

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2021 Institut Pasteur                                       #
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


import modeller


if __name__ == '__main__':
    # See: https://salilab.org/modeller/wiki/Missing%20residues
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='Transfer sequence from ref to target by keeping coordinates of equivalent atoms. Atoms which do not have an equivalent are built based on the internal coordinates specified in the residue topology library')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-t', '--target', help='mobile pdb structure file to transfer sequence on', type=str)
    parser.add_argument('-r', '--ref', help='reference pdb structure file with sequence to transfer', type=str)
    args = parser.parse_args()

    env = modeller.environ()
    lib = '/usr/lib/modeller9.23/modlib'
    env.libs.topology.read(file=f'{lib}/top_heav.lib')
    env.libs.parameters.read(file=f'{lib}/par.lib')
    aln = modeller.alignment(env)

    target = modeller.model(env, file=args.target)
    target_name = args.target.split('.')[0]
    aln.append_model(target, align_codes=target_name)

    ref = modeller.model(env, file=args.ref)
    ref_name = args.ref.split('.')[0]
    aln.append_model(ref, align_codes=ref_name)

    aln.align()
    # aln.align3d()
    alnfile = f'{target_name}_{ref_name}.seq'
    aln.write(file=alnfile)

    mdl = modeller.model(env)
    mdl.generate_topology(aln[ref_name])
    # Assign the average of the equivalent template coordinates to MODEL:
    mdl.transfer_xyz(aln)
    # Get the remaining undefined coordinates from internal coordinates:
    mdl.build(initialize_xyz=False, build_method='INTERNAL_COORDINATES')

    # transfer residue numbers and chain ids from ref to target
    # ref first
    # mdl last
    aln = modeller.alignment(env)
    aln.append_model(ref, align_codes=ref_name)
    aln.append_model(mdl, align_codes='model')
    mdl.res_num_from(ref, aln)

    mdl.write(file=f'{target_name}_{ref_name}.pdb')

    # env = modeller.environ()
    # a = automodel(env, alnfile=alnfile,
    #               knowns=target_name, sequence=ref_name)
    # # a = MyModel(env, alnfile=alnfile, knowns=target_name, sequence=ref_name)
    # a.starting_model = 1
    # a.ending_model = 1
    # # a.max_var_iterations = 10
    # # a.md_level = refine.very_fast
    # a.make()
    # cleanup(ref_name)
