#!/usr/bin/env sh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2017-12-04 11:00:53 (UTC+0100)

# Make a squashfs archive of the file

ARCHIVE_DIR="."

INPATH=$1

BASENAME=$(basename "$INPATH")
OUTPATH="$ARCHIVE_DIR/$BASENAME.sqsh"

if [ ! -f $OUTPATH ]; then
    mksquashfs $INPATH $OUTPATH && echo "archive $OUTPATH successfully created"
else
    (>&2 echo "$OUTPATH already exists. Cannot create archive...")
fi
