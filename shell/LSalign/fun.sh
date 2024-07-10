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

test_LSalign_smi () {
    SMI1="CCOC1=CC(C(O)C1O)n1cnc2c(N)ncnc12"
    SMI2="Cc1ccccc1NC(=O)C(C1CCCCC1)n1c(nc2cc(F)c(F)cc12)-c1ccc(Cl)cc1"
    ./LSalign_smi.sh --smi1 $SMI1 --smi2 $SMI2
}
