#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Jul  1 11:11:58 2025

# Print the usage of the script if no data is provided in standard input
if [ -t 0 ]; then
    echo "Usage: cat <fasta_file> | $0"
    echo "This script computes the reverse complement of a FASTA file."
    exit 1
fi

reverse_complement() {
    awk '{
        if ($0 ~ /^>/){
            # print the fasta header
            if ($0 ~ "strand=-"){
                modified_header = gensub(/strand=-/, "strand=+", "g", $0)
                print modified_header
            }
            else{
                if ($0 ~ "strand=+"){
                    modified_header = gensub(/strand=+/, "strand=-", "g", $0)
                    print modified_header
                }
                else{
                    print $0
                }
            }
        }
        else{
            seq = seq $0
        }
    }
    END{
        # Inversion de la séquence
        rev = ""
        for (i=length(seq); i>0; i--) rev=rev substr(seq,i,1)
        # print rev
        system("echo "rev"|tr 'ACGTacgt' 'TGCAtgca'")
    }'
}

reverse_complement
