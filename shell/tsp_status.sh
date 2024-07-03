#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jun  6 09:07:06 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Help message
    -h, --help print this help message and exit
EOF
}

N=1  # Default value
while [ "$#" -gt 0 ]; do
    case $1 in
        -n|--number) N="$2"; shift ;;
        -h|--help) usage; exit 0 ;;
        --) OTHER="${@:2}";break; shift;;  # Everything after the '--' symbol
        *) usage; exit 1 ;;
    esac
    shift
done

tsp | awk 'NR>1{
            status=$2
            n[status]+=1
            if (status=="finished"){
                split($5,a,"/")
                t+=a[1]
                if ($4*1>0){
                  n_error+=1
                }
                else{
                  n_error+=0
                }
            }
           }
           END{
            for (status in n){
                print status,n[status]
            }
            print "n_error", n_error
            if (n["finished"]>0){
              t_mean=t/n["finished"]
              system("echo -n \x27average_running_time \x27; qalc -t "t_mean"s")
              if (n["running"]>0){
                eta=((n["queued"]+n["running"])/n["running"])*t_mean
                system("echo -n \x27ETA \x27; qalc -t "eta"s")
              }
            }
            n_total=n["queued"]+n["running"]+n["finished"]
            if (n_total>0){
              progress=n["finished"]/(n_total)
              print("progress "progress*100" %")
            }
            if (n["finished"]==1000){
              system("tsp -C")
            }
           }' > $MYTMP/tsp_status.out
cat $MYTMP/tsp_status.out
