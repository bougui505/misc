#!/usr/bin/env python3
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-09 15:23:41 (UTC+0200)

import sys
import argparse

parser = argparse.ArgumentParser(description='Column ruler for the shell')
parser.add_argument('-g', '--glob', action='store_true', default=False,
                    help='Do not restart numbering at each line')
parser.add_argument('-c', '--color', action='store_true', default=False,
                    help='Print with color. termcolor is required')
args = parser.parse_args()

if args.color:
    from termcolor import colored, cprint

def printer(instr):
    if args.color:
        cprint(instr, 'cyan')
    else:
        print(instr)


def get_rulers(strlen):
    ruler1_ = '....^....|'
    ruler2_ = '1234567890'
    r1len = len(ruler1_)
    ruler1 = ''
    ruler2 = ''
    for i in range((strlen + r1len) // r1len):
        ruler1 += f'...{i:03d}...|'
        ruler2 += ruler2_
    ruler1 = ruler1[:strlen]
    ruler2 = ruler2[:strlen]
    ruler1 += f'-> {len(ruler1)}'
    return ruler1, ruler2


with sys.stdin as inpipe:
    lines = []
    linelen = []
    for line in inpipe:
        line = line.strip()
        if args.glob:
            lines.append(line)
            linelen.append(len(line))
        else:
            sys.stdout.write(line)
            sys.stdout.write("\n")
            strlen = len(line)
            ruler1, ruler2 = get_rulers(strlen)
            printer(ruler1)
            printer(ruler2)
if args.glob:
    lines = ''.join(lines)
    strlen = len(lines)
    ruler1, ruler2 = get_rulers(strlen)
    i = 0
    for n in linelen:
        print(lines[i:i + n])
        ruler1__ = ruler1[i:i + n] + f'-> {i + n}'
        printer(ruler1__)
        printer(ruler2[i:i + n])
        print("")
        i += n
