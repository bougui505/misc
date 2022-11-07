#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2018-09-18 17:19:45 (UTC+0200)

DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
BMFDIR=$HOME/misc
FILENAME=$1:a

LIGHTGREEN='\033[1;32m'
LIGHTRED='\033[1;31m'
NC='\033[0m' # No Color

####
cd $BMFDIR &&
/home/bougui/bin/tidyup
ERROR=$(ln -s $FILENAME . 2>&1) &&
touch -h $FILENAME:t &&
printf "${LIGHTGREEN}$FILENAME linked in $BMFDIR ${NC}\n" ||
printf "${LIGHTRED}bmf error | $ERROR ${NC}\n"
