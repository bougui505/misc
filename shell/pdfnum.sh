#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon Mar 25 11:39:31 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
    -h, --help print this help message and exit
    -i, --inp input pdf file
EOF
}

while [ "$#" -gt 0 ]; do
    case $1 in
        -i|--input) INP="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

NPAGES=$(pdftk "$INP" dump_data | grep "NumberOfPages" | cut -d":" -f2 | tr -d " ")
echo "NPAGES=$NPAGES"
PAGESIZE=$(pdfinfo $INP | grep "^Page size:" | awk -F":" '{print $2}' | tr -d " ")
echo "PAGESIZE=$PAGESIZE"
WIDTH=$(echo $PAGESIZE | awk -F"x" '{print $1}')
echo "WIDTH=$WIDTH"
HEIGHT=$(echo $PAGESIZE | awk -F"x" '{print $2}' | tr -d 'abcdefghijklmnopqrtsuvwxyz')
echo "HEIGHT=$HEIGHT"
XPOS=$(seq 1 | awk -v w=$WIDTH '{print w*0.98}')
echo "XPOS=$XPOS"
YPOS=$(seq 1 | awk -v h=$HEIGHT '{print h*0.02}')
echo "YPOS=$YPOS"
for i in $(seq $NPAGES); do

    cat << EOF > $MYTMP/$i.svg
<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<svg
   width="$WIDTH"
   height="$HEIGHT"
   viewBox="0 0 $WIDTH $HEIGHT"
   version="1.1"
   id="svg5"
   xmlns="http://www.w3.org/2000/svg"
   xmlns:svg="http://www.w3.org/2000/svg">
  <defs
     id="defs2" />
  <g
     id="layer1" />
    <text
     xml:space="preserve"
     style="font-size:18.6667px;line-height:1.25;font-family:'DejaVuSansM Nerd Font Mono';-inkscape-font-specification:'DejaVuSansM Nerd Font Mono';text-align:center;text-decoration-color:#000000;letter-spacing:0px;word-spacing:0px;text-anchor:middle;fill:#000000;fill-opacity:0.846172;stroke:none;stroke-linecap:round;stroke-linejoin:round;stroke-miterlimit:10;stop-color:#000000"
     x="1406.4586"
     y="793.44873"
     id="text234"><tspan
       id="tspan232"
       x="$XPOS"
       y="$YPOS"
       style="font-style:normal;font-variant:normal;font-weight:normal;font-stretch:normal;font-size:18.6667px;font-family:monospace;-inkscape-font-specification:monospace">$i/$NPAGES</tspan></text>
</svg>
EOF
    inkscape $MYTMP/$i.svg --export-type=pdf --export-filename=$MYTMP/$i.pdf
done
OUT=${INP:r}_num.pdf
echo "OUT=$OUT"
# pagenum=$(pdftk "$INP" dump_data | grep "NumberOfPages" | cut -d":" -f2)
# enscript -r -M A4 -L1 --header='$%/$=' --output - < <(for i in $(seq "$pagenum"); do echo; done) | ps2pdf - | pdftk "$INP" multistamp - output $output
pdftk $(ls -v $MYTMP/*.pdf) cat output $MYTMP/pdfnum.pdf
pdftk "$INP" multistamp $MYTMP/pdfnum.pdf output $OUT
