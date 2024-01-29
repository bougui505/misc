#!/usr/bin/env bash

PERIOD=15  # PERIOD in minutes
PERIOD=$(qalc -t "$PERIOD min to s" | awk '{print $1}')
OUTDIR="/media/usb0/t-temp_c-humidity"
OUTFILE="$OUTDIR/data.dat"
NLINES=10000  # maximum number of measurement to keep

echo "PERIOD=$PERIOD (s)"
echo "OUTFILE=$OUTFILE"

if pgrep pigpiod; then
    echo "pigpiod is running"
else
    sudo pigpiod
fi

if [[ ! -d $OUTDIR ]]; then
    echo "$OUTDIR does not exist"
    exit 1
fi

while sleep $PERIOD; do
    touch $OUTFILE
    tail -n $NLINES $OUTFILE | sponge $OUTFILE
    ./temperature_sample.py >> $OUTFILE
    ./temperature_plotter.py
    # outputs:
    # seconds since epoch temperature humidity
done
