#!/usr/bin/env zsh

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2024 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Tue Sep 24 14:40:47 2024

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwritting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
trap 'rm -rf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

MAILDIR="/home/bougui/mails/mutt"
touch $MAILDIR/mail.list

rga-fzf() {
    cd $MAILDIR/gzips
    RG_PREFIX="rga --files-with-matches --rga-adapters=+mail"
    local file
    query_file="$(
        FZF_DEFAULT_COMMAND="$RG_PREFIX '$1'" \
            fzf --multi --color=light --sort --preview="[[ ! -z {} ]] && rga --rga-adapters=+mail --pretty --context 30 {q} {}" \
            --phony -q "$1" \
            --bind "change:reload:$RG_PREFIX {q}" \
            --preview-window="70%:wrap")"
    cd -
    for x in $(echo $query_file); do
        fullpath="$MAILDIR/gzips/$x"
        echo $fullpath
        # zless $fullpath
        zcat $fullpath | batcat --decorations never --color always -l email | /bin/less -R
    done
}

find $MAILDIR/backup -type f > $MYTMP/mail.list

gawk '{
if (ARGIND==1){
    DONE[$0]=$0
}
if (ARGIND==2){
    if (!($0 in DONE)){
        print $0
    }
}
}' $MAILDIR/mail.list $MYTMP/mail.list > $MYTMP/todo.list

for mail in $(cat $MYTMP/todo.list); do
    GZIPDIR=$MAILDIR/gzips/$(echo "${mail:h}"|sed "s,$MAILDIR/,,")
    mkdir -p $GZIPDIR
    GZIPFILE=$GZIPDIR"/$mail:t.gz"
    echo "Creating: $GZIPFILE"
    # if grep -Fq "Content-Type: text/plain" $mail; then
    #     mu view $mail | gzip > "$GZIPFILE" \
    #         && echo $mail >> $MAILDIR/mail.list
    # elif grep -Fq "Content-Type: text/html" $mail; then
    #     decode_email -e $mail | gzip > $GZIPFILE \
    #         && echo $mail >> $MAILDIR/mail.list
    # else
    #     mu view $mail | gzip > "$GZIPFILE" \
    #         && echo $mail >> $MAILDIR/mail.list
    # fi
    decode_email -e $mail | gzip > $GZIPFILE \
        && echo $mail >> $MAILDIR/mail.list
done

rga-fzf $1
