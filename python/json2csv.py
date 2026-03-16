#!/usr/bin/env python3
#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2026 Institut Pasteur                                       #
#############################################################################
#
# creation_date: 2026-02-16

import pandas as pd
import sys
import json

def load_json_data(jsonfile):
    if jsonfile == '-':
        data = json.load(sys.stdin)
    else:
        with open(jsonfile, 'r') as f:
            data = json.load(f)
    return data

if __name__ == "__main__":
    if len(sys.argv) > 1:
        jsonfile = sys.argv[1]
    else:
        jsonfile = '-'
    
    out = sys.stdout
    data = load_json_data(jsonfile)
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    df.to_csv(out)
