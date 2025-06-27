#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Jun 27 09:42:24 2025

# set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script
# 
# function usage () {
#     cat << EOF
# Help message
#     -h, --help print this help message and exit
# EOF
# }
# 
# N=1  # Default value
# while [ "$#" -gt 0 ]; do
#     case $1 in
#         -n|--number) N="$2"; shift ;;
#         -h|--help) usage; exit 0 ;;
#         --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
#         *) usage; exit 1 ;;
#     esac
#     shift
# done

# adapted from: https://askubuntu.com/a/942735/415396

TARGET_SITE="$1"
PDF_DIR="$(echo "$TARGET_SITE" | sed -e 's/https\?:\/\///' -e 's,/,_,g')-pdfs"
OUTPDF="$(echo $TARGET_SITE | sed -e 's/https\?:\/\///' -e 's,/,_,g').pdf"
NPROC=$(nproc)

mkdir -p "$PDF_DIR"
echo "PDF files written in $PDF_DIR"
echo "out PDF file: $OUTPDF"

wget --spider --force-html -r -l2 "$TARGET_SITE" 2>&1 \
  | grep '^--' \
  | awk '{ print $3 }' \
  | grep -v '\.\(css\|js\|png\|gif\|jpg\|txt\)$' \
  > url-list.txt
NURL=$(wc -l url-list.txt | awk '{print $1}')
COUNT=0
# for i in $(cat url-list.txt); do
#   # make a progress bar with a percent of advancement
#   ((++COUNT))
#   PERCENT=$(echo "scale=2; $COUNT/$NURL * 100" | bc)
#   printf "\rProcessing %s (%d/%d) %.2f%%                                 " "$i" "$COUNT" "$NURL" "$PERCENT"
#   wkhtmltopdf "$i" "$PDF_DIR/$(echo "$i" | sed -e 's/https\?:\/\///' -e 's/\//-/g' ).pdf" 2>&1 &> /dev/null
# done
# rewrite the loop above in parallel
parallel --bar --eta -j $NPROC wkhtmltopdf --log-level none --lowquality {} "$PDF_DIR/{#}.pdf" :::: url-list.txt

gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -dPDFSETTINGS=/prepress -sOutputFile=$OUTPDF $(ls -v -1 $PDF_DIR/*.pdf)
