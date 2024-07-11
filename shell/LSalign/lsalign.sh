#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Wed Jul 10 16:46:56 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Run flexible LSalign algorithm (see: https://zhanggroup.org/LS-align/) on 2 molecular SMILES
    -h, --help print this help message and exit
    --smi1 first SMILES
    --smi2 second SMILES
    --rec rec file to process. If given, '--smi1' give the field name for the first SMILES and
          '--smi2' the second one.
EOF
}

N=1  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        --smi1) SMI1="$2"; shift ;;
        --smi2) SMI2="$2"; shift ;;
        --rec) REC="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

exit1 (){
    echo -1
    exit 1
}

if [[ -z $REC ]]; then
    $DIRSCRIPT/_LSalign_smi_.sh $SMI1 $SMI2
else
    OUTREC=${REC:r:r}_lsalign.rec.gz
    $DIRSCRIPT/../../python/rec.py --file $REC \
                                   --fields $SMI1 $SMI2 \
                                   --run PC-score1,PC-score2,PC-score_max,Pval1,Pval2,jaccard,rmsd,size1,size2=$DIRSCRIPT/_LSalign_rec_.sh \
        | gzip > $OUTREC
fi
