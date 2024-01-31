#!/usr/bin/env bash

OUTDIR="/media/usb0/t-temp_c-humidity"
OUTFILE="$OUTDIR/data.dat"
NLINES=10000  # maximum number of measurement to keep

DIRSCRIPT="$(dirname "$(readlink -f "$0")")"

date

echo "DIRSCRIPT=$DIRSCRIPT"
echo "OUTFILE=$OUTFILE"

if pgrep pigpiod; then
    echo "pigpiod is running"
else
    echo "pigpiod is not runningi. Please run:"
    echo "sudo pigpiod"
    exit 1
fi

if [[ ! -d $OUTDIR ]]; then
    echo "$OUTDIR does not exist"
    exit 1
fi

touch $OUTFILE
tail -n $NLINES $OUTFILE | awk 'NF==3{print}' | sponge $OUTFILE
$DIRSCRIPT/temperature_sample.py | awk 'NF==3{print}' >> $OUTFILE
$DIRSCRIPT/temperature_plotter.py
# outputs:
# seconds since epoch temperature humidity
