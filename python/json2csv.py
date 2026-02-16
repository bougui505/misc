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

if __name__ == "__main__":
    jsonfile = sys.argv[1]
    out = sys.stdout
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    df = pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame([data])
    df.to_csv(out)
