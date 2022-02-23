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

import numpy as np


def create_filetest(nline, maxwords=20, outfilename='test.txt'):
    with open('test.txt', 'w') as f:
        for i in range(nline):
            nwords = np.random.randint(1, high=maxwords)
            outstr = 'STARTLINE '
            outstr += f'{i} ' * nwords
            outstr += 'ENDLINE'
            outstr += '\n'
            f.write(outstr)


class Fileseek(object):
    """
    >>> create_filetest(100)
    >>> fileseek = Fileseek('test.txt')
    >>> fileseek.getline(31)
    'STARTLINE 31 ... ENDLINE'
    >>> fileseek.getline(10)
    'STARTLINE 10 ... ENDLINE'
    """
    def __init__(self, filename):
        self.file = open(filename, 'rb')
        self.mapping = self.maplines()

    def maplines(self):
        self.file.seek(0)
        offset = 0
        mapping = {}
        for line_id, line in enumerate(self.file):
            mapping[line_id] = offset
            offset += len(line)
        self.file.seek(0)
        return mapping

    def getline(self, lineid):
        offset = self.mapping[lineid]
        self.file.seek(offset)
        line = self.file.readline()
        return line.decode().strip()


if __name__ == '__main__':
    import sys
    import doctest
    import argparse
    # argparse.ArgumentParser(prog=None, usage=None, description=None, epilog=None, parents=[], formatter_class=argparse.HelpFormatter, prefix_chars='-', fromfile_prefix_chars=None, argument_default=None, conflict_handler='error', add_help=True, allow_abbrev=True, exit_on_error=True)
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
