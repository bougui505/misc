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
import os


def alnstr(aln, seqlen=None, strlen=80, threshold=0.5):
    """
    >>> seq1 = range(200)
    >>> seq2 = range(150, 350)
    >>> aln = dict(zip(seq1, seq2))
    >>> for i in range(30,70):
    ...     aln[i] = None
    >>> aln
    {0: 150, 1: 151, 2: 152, 3: 153, 4: 154, 5: 155, 6: 156, 7: 157, 8: 158, 9: 159, 10: 160, 11: 161, 12: 162, 13: 163, 14: 164, 15: 165, 16: 166, 17: 167, 18: 168, 19: 169, 20: 170, 21: 171, 22: 172, 23: 173, 24: 174, 25: 175, 26: 176, 27: 177, 28: 178, 29: 179, 30: None, 31: None, 32: None, 33: None, 34: None, 35: None, 36: None, 37: None, 38: None, 39: None, 40: None, 41: None, 42: None, 43: None, 44: None, 45: None, 46: None, 47: None, 48: None, 49: None, 50: None, 51: None, 52: None, 53: None, 54: None, 55: None, 56: None, 57: None, 58: None, 59: None, 60: None, 61: None, 62: None, 63: None, 64: None, 65: None, 66: None, 67: None, 68: None, 69: None, 70: 220, 71: 221, 72: 222, 73: 223, 74: 224, 75: 225, 76: 226, 77: 227, 78: 228, 79: 229, 80: 230, 81: 231, 82: 232, 83: 233, 84: 234, 85: 235, 86: 236, 87: 237, 88: 238, 89: 239, 90: 240, 91: 241, 92: 242, 93: 243, 94: 244, 95: 245, 96: 246, 97: 247, 98: 248, 99: 249, 100: 250, 101: 251, 102: 252, 103: 253, 104: 254, 105: 255, 106: 256, 107: 257, 108: 258, 109: 259, 110: 260, 111: 261, 112: 262, 113: 263, 114: 264, 115: 265, 116: 266, 117: 267, 118: 268, 119: 269, 120: 270, 121: 271, 122: 272, 123: 273, 124: 274, 125: 275, 126: 276, 127: 277, 128: 278, 129: 279, 130: 280, 131: 281, 132: 282, 133: 283, 134: 284, 135: 285, 136: 286, 137: 287, 138: 288, 139: 289, 140: 290, 141: 291, 142: 292, 143: 293, 144: 294, 145: 295, 146: 296, 147: 297, 148: 298, 149: 299, 150: 300, 151: 301, 152: 302, 153: 303, 154: 304, 155: 305, 156: 306, 157: 307, 158: 308, 159: 309, 160: 310, 161: 311, 162: 312, 163: 313, 164: 314, 165: 315, 166: 316, 167: 317, 168: 318, 169: 319, 170: 320, 171: 321, 172: 322, 173: 323, 174: 324, 175: 325, 176: 326, 177: 327, 178: 328, 179: 329, 180: 330, 181: 331, 182: 332, 183: 333, 184: 334, 185: 335, 186: 336, 187: 337, 188: 338, 189: 339, 190: 340, 191: 341, 192: 342, 193: 343, 194: 344, 195: 345, 196: 346, 197: 347, 198: 348, 199: 349}
    >>> alnstr(aln, strlen=30)
    '*****-------*********************'
    """
    if seqlen is None:
        seqlen = max(aln.keys())
    strlen = min(strlen, seqlen)
    binlen = seqlen // strlen
    binindex = 0
    bincount = 0
    out = ''
    for i in range(seqlen):
        binindex += 1
        if i in aln:
            if aln[i] is not None:
                bincount += 1
        if binindex >= binlen:
            if bincount / binlen >= threshold:
                out += '*'
            else:
                out += '-'
            binindex = 0
            bincount = 0
    return out


def log(msg):
    try:
        logging.info(msg)
    except NameError:
        pass


def GetScriptDir():
    scriptpath = os.path.realpath(__file__)
    scriptdir = os.path.dirname(scriptpath)
    return scriptdir


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
    parser.add_argument('-a', '--arg1')
    parser.add_argument('--test', help='Test the code', action='store_true')
    args = parser.parse_args()

    if args.test:
        doctest.testmod(optionflags=doctest.ELLIPSIS | doctest.REPORT_ONLY_FIRST_FAILURE)
        sys.exit()
