#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-10-09 13:19:40 (UTC+0200)

ruler() {
    # [ ! -t 0 ] && echo "stdin has data" || echo "stdin is empty" # (See: https://unix.stackexchange.com/a/388462/68794)
    [ ! -t 0 ] && read PIPEIN && echo $PIPEIN
    [ ! -t 0 ] && STRLEN=${#PIPEIN} || STRLEN=$COLUMNS
    s='....^....|'
    w=${#s}; str=$( for (( i=1; $i<=$(( ($STRLEN + $w) / $w )) ; i=$i+1 )); do
                        DEC=$(printf "%03d" $i-1)
                        echo -n "...$DEC...|"
                    done )
    str=$(echo $str | cut -c -$STRLEN)
    echo $str

    s='1234567890'
    w=${#s}; str=$( for (( i=1; $i<=$(( ($STRLEN + $w) / $w )) ; i=$i+1 )); do echo -n $s; done )
    str=$(echo $str | cut -c -$STRLEN)
    echo $str
    [ ! -t 0 ] && OUTLEN="String length: $STRLEN" && printf "%*s\n" $(((${#title}+$COLUMNS)/2)) "$OUTLEN"
}
