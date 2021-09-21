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

import numpy as np
import scipy.spatial.distance as distance
import scipy.optimize as optimize


def merge_lists(list1, list2):
    """
    Merge list1 and list2 into a list with the most evenly spaced items,
    but keeping the original order of list1 and list2

    Args:
        path1: list of tuples (e.g. [(1, 1, 1), (2, 2, 2)]) or list of values
        path2: list of tuples (e.g. [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3)]) or list of values
    Returns:

    >>> merge_lists([1, 3, 5], [0, 1, 2, 4])
    [0, 1, 2, 3, 4, 5]

    >>> merge_lists([0, 3], [1, 2, 3, 4])
    [0, 1, 2, 3, 4]

    >>> merge_lists([1, 2, 3, 4], [0, 3])
    [0, 1, 2, 3, 4]

    >>> merge_lists([(1, 1, 1), (3, 3, 3), (5, 5, 5)], [(0, 0, 0), (1, 1, 1), (2, 2, 2), (4, 4, 4)])
    [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]

    """
    if len(list1) > len(list2):
        list1, list2 = list2, list1
    list1 = list1[::-1]
    list2 = list2[::-1]
    arr1 = np.asarray(list1)
    arr2 = np.asarray(list2)
    assert arr1.ndim == arr2.ndim
    if arr1.ndim == 1:
        arr1 = arr1[:, None]
        arr2 = arr2[:, None]
    dmat = distance.cdist(arr1, arr2)
    row_inds, col_inds = optimize.linear_sum_assignment(dmat)
    out = []
    while len(list1) > 0 or len(list2) > 0:
        # i = len(list1) - 1
        j = len(list2) - 1
        if j not in col_inds:
            out.append(list2.pop())
        else:
            col_ind = j
            row_ind = row_inds[col_inds == j]
            if dmat[row_ind, col_ind] == 0:
                out.append(list2.pop())
                list1.pop()
            else:
                e1, e2 = list1.pop(), list2.pop()
                if e2 > e1:
                    e1, e2 = e2, e1
                out.extend([e2, e1])
    out = np.squeeze(np.asarray(out))
    out = [tuple(e) if hasattr(e, '__iter__') else e for e in out]
    return out


if __name__ == '__main__':
    import doctest
    doctest.testmod()
