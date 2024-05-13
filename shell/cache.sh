#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu May  2 10:29:09 2024

# set -e  # exit on error
# set -o pipefail  # exit when a process in the pipe fails
# set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script
trap 'rm -rf .header .cachefile .cksum' EXIT INT

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
    -n, --nlines number of header lines to compute the md5sum on (default: 1000)
EOF
}

N=1000
while [ "$#" -gt 0 ]; do
    case $1 in
        -h|--help) usage; exit 0 ;;
        -n|--nlines) N=$2; shift;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

cat \
    | awk -v "N=$N" '{
        if (NR<=N){
            print > ".header"
        }
        if (NR==N){
            # flush the buffer
            fflush()
            # Compute the md5sum checksum for the header
            "md5sum -z .header"|getline cksum
            split(cksum,a," ")
            cksum=a[1]
            print(cksum) > ".cksum"
            # check if cache file exists
            "ls .cache/"cksum|getline f
            if (f==".cache/"cksum){
                exit 0
            }
        }
        print
    }' \
    2> >(grep -v "^ls:") \
    | gzip > .cachefile

CKSUM=$(cat .cksum)
CACHEFILE=".cache/${CKSUM}"
[[ ! -d .cache ]] && mkdir .cache
if [[ ! -f $CACHEFILE ]]; then
    echo "#CACHEFILE=$CACHEFILE" | gzip > $CACHEFILE
    echo "#DATE=$(date +'%Y/%m/%d %H:%M:%S')" | gzip >> $CACHEFILE
    cat .cachefile >> $CACHEFILE
fi
$DIRSCRIPT/pcat.sh $CACHEFILE | zcat
