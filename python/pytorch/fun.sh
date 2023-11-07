#!/usr/bin/env bash

Cyan='\033[0;36m'
NC='\033[0m' # No Color

echo -e "$Cyan"
cat << EOF
$(date): sourcing $0
EOF
echo -e $FUNFILES$NC

MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap '/bin/rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

[ -z $FUNFILES ] && export FUNFILES=""
precmd() {
    [ -f fun.sh ] && source fun.sh && FUNFILES+="$(realpath fun.sh)" && FUNFILES=$(echo $FUNFILES | awk -F":" '{for (i=1;i<=NF;i++){print $i}}' | sort -u | awk '{printf $1":"}')
}

test_MDS () {
    sing ./MDS.py --rec data/mds_inp.rec.gz distance --niter 50000
}

test_MDS_batch () {
    sing ./MDS.py --rec data/mds_inp.rec.gz distance --niter 50000 -bs 50 --nepochs 100 --min_delta_epoch 0.1
}

test_MDS_pts () {
    sing ./MDS_pts.py --rec data/mds_pts.rec.gz --distance distance.py getdist --niter 50000 -bs 50 --nepochs 100  # --min_delta_epoch 0.1
}

plot_MDS () {
    rec --file data/mds.rec.gz --fields mds_i class_i | grep -v "^#" | uniq | tr -d '[' | tr -d ']' | tr -d "," | plot2 --scatter --fields x y z --colorbar
}

plot_MDS_pts () {
    rec --file data/mds_pts_out.rec.gz --fields mds class_i| tr -d '[' | tr -d ']' | tr -d ',' | plot2 --scatter --fields x y z
}
