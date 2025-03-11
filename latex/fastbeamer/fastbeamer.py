#!/usr/bin/env python3
# -*- coding: UTF8 -*-

import glob
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
import os
import subprocess
import time

from PyPDF2 import PdfFileReader, PdfMerger

from misc import hashfile


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


def get_header(texfilename):
    """
    >>> h = get_header('test.tex')
    >>> print(h)
    \\documentclass[aspectratio=169]{beamer}
    \\usepackage{beamer_header}
    \\title{My title}
    \\begin{document}
    <BLANKLINE>
    """
    header = ""
    with open(texfilename, 'r') as texfile:
        for line in texfile:
            header += line
            if line.startswith("\\begin{document}"):
                break
    return header


def get_footer(texfilename):
    """
    >>> f = get_footer("test.tex")
    >>> print(f)
    \\appendix
    \\begin{frame}[allowframebreaks]{References}
    \\bibliography{biblio}
    \\bibliographystyle{apalike}
    \\end{frame}
    \\end{document}
    <BLANKLINE>
    """
    footer = ""
    isfooter = False
    with open(texfilename, 'r') as texfile:
        for line in texfile:
            if line.startswith("\\appendix"):
                isfooter = True
            if isfooter:
                footer += line
    if footer=="":
        footer="\\end{document}"
    return footer


def splitframes(texfilename, header=None, footer=None):
    """
    # >>> splitframes('test.tex')
    """
    outdir = ".fastbeamer"
    cwd = os.getcwd()
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    if not os.path.islink(f'{outdir}/beamer_header.sty'):
        os.chdir(outdir)
        os.symlink('../beamer_header.sty', 'beamer_header.sty')
        os.chdir(cwd)
    if not os.path.islink(f'{outdir}/biblio.bib'):
        os.chdir(outdir)
        os.symlink('../biblio.bib', 'biblio.bib')
        os.chdir(cwd)
    if os.path.isdir('figures'):
        if not os.path.islink(f'{outdir}/figures'):
            os.chdir(outdir)
            os.symlink('../figures', 'figures')
            os.chdir(cwd)
    i = -1
    outfile = None
    with open(texfilename, 'r') as texfile:
        for line in texfile:
            if line.startswith("\\appendix"):
                break
            if line.startswith("\\begin{frame}"):
                i += 1
                outfile = open(f'{outdir}/slide_{i}.tex', 'w')
                if header is not None:
                    outfile.write(header)
            if line.startswith("\\end{frame}"):
                outfile.write("\\end{frame}\n")
                if footer is not None:
                    outfile.write(footer)
                outfile.close()
                outfile = None
            if outfile is not None:
                outfile.write(line)
    n = i + 1
    print(f'Total number of slides: {n}')
    # Remove deleted slides
    all_slides = set(glob.glob(f"{outdir}/slide_*.tex"))
    current_slides = [f"{outdir}/slide_{i}.tex" for i in range(n)]
    torm = all_slides - set(current_slides)
    for slide in torm:
        print(f'Deleting old slide: {slide}')
        os.remove(slide)
    return current_slides


class Fastbeamer(object):
    def __init__(self, texfilename):
        """
        >>> fb = Fastbeamer('test.tex')
        Total number of slides: 2
        """
        self.texfilename = texfilename
        self.hash = None
        isopened = False
        p = subprocess.Popen('pwd', shell=True)
        while True:
            if hashfile.cathash([self.texfilename, 'figures']) != self.hash:
                self.hash = hashfile.cathash([self.texfilename, 'figures'])
                self.header = get_header(self.texfilename)
                self.footer = get_footer(self.texfilename)
                self.slides = splitframes(self.texfilename, header=self.header, footer=self.footer)
                self.cwd = os.getcwd()
                self.pdfs = self.compile()
                self.merge()
                # see: https://stackoverflow.com/a/43276598/1679629 for p.poll()
                if os.path.exists('fb-build/fastbeamer.pdf') and p.poll() is not None:
                    p = subprocess.Popen('evince fb-build/fastbeamer.pdf', shell=True)
                    isopened = True
            time.sleep(1)

    def compile(self):
        outpdf = []
        processes = []
        for slide in self.slides:
            # use call for non-parallel compilation
            p = subprocess.Popen(f'latexmk -shell-escape -pdf -outdir=fb-build {slide}', shell=True)
            processes.append(p)
            basename = os.path.splitext(os.path.basename(slide))[0]
            outpdf.append(f'fb-build/{basename}.pdf')
        # p_status = p.wait()
        p_status = [p.wait() for p in processes]
        outpdf = [e for e in outpdf if os.path.exists(e)]
        return outpdf

    def merge(self):
        """
        """
        merger = PdfMerger()
        for pdf in self.pdfs:
            print(f"merging pdf {pdf}")
            with open(pdf, 'rb') as pdffile:
                readpdf = PdfFileReader(pdffile)
                totalpages = readpdf.numPages
                merger.append(fileobj=pdffile, pages=(0, totalpages-1))  # (-1 is to remove the slides of references -- bibliography)
        with open('fb-build/fastbeamer.pdf', 'wb') as output:
            merger.write(output)
        merger.close()


if __name__ == '__main__':
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
    parser = argparse.ArgumentParser(description='')
    # parser.add_argument(name or flags...[, action][, nargs][, const][, default][, type][, choices][, required][, help][, metavar][, dest])
    parser.add_argument('-i', '--inp', help='Input tex beamer file to compile')
    parser.add_argument('--test', help='Test the code', action='store_true')
    parser.add_argument('--func', help='Test only the given function(s)', nargs='+')
    args = parser.parse_args()

    # If log is present log the arguments to the log file:
    for k, v in args._get_kwargs():
        log(f'# {k}: {v}')

    if args.test:
        if args.func is None:
            doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        else:
            for f in args.func:
                print(f'Testing {f}')
                f = getattr(sys.modules[__name__], f)
                doctest.run_docstring_examples(f, globals())
        sys.exit()
    if args.inp is not None:
        fb = Fastbeamer(args.inp)
