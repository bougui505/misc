#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Thu Jan  8 16:32:33 2026

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

{
echo "Backuped directories:"
echo "====================="
sudo grep "include" /root/.config/backintime/config | grep value | cut -d'=' -f2
echo "====================="

echo "snapshots-list:"
echo "==============="
sudo backintime snapshots-list
echo "==============="

echo "journalctl logs:"
echo "================"
sudo journalctl _UID=0 -t backintime | tail -n 15
echo "================"

# echo "check-config:"
# echo "============="
# sudo backintime check-config
# echo "============="
} > $MYTMP/report.txt

cat $MYTMP/report.txt
