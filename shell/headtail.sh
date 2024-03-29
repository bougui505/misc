#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-11-16 10:03:08 (UTC+0100)

usage ()
{
	echo "Usage"
	echo "headtail filename"
	echo "Print the first and the last 10 lines of the file"
	echo "    -h print this help message and exit"
	echo "    -n NUM    print the first and the last NUM lines"
	echo "    -d        Do not print the delimiter '[...]' between the head
			    and the tail of the file"
}

# if [ $# -lt 1 ]; then
# 	usage
# 	exit
# fi

N=10
DELIM=1
for i in "$@"; do
	case $i in
		"-h")
			usage
			exit
			;;
		"-n")
			shift # shift the counter of $1 variables ($1 becomes $2) and so on for $2 ...
			N=$1
			shift
			;;
		"-d")
			DELIM=0
			shift
			;;
	esac
done
FILENAME=$1

awk -v N=$N -v DELIM=$DELIM '{
data[NR]=$0
} 
END{
	for (j=1; j<=N; j++){
		if (j<=NR){
			print data[j]
		}
	}
	if (DELIM==1 && j<=NR && 2*N<NR){print "[...]"}
	for (k=NR-N+1;k<=NR;k++){
		if (k>0 && k>=j){
			print data[k]
		}
	}
}' $FILENAME
