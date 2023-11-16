#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr
# https://research.pasteur.fr/en/member/guillaume-bouvier/
# 2020-05-29 23:23:44 (UTC+0200)

func runcmd () {
    OUTPUT=$(eval $1)
    echo "\`\`\`"
    echo "$ $1\n"
    echo "$OUTPUT"
    echo "\`\`\`"
}

cat << EOF
# npcli
Feed stdin data to a numpy array (default variable name is \`A\`) and apply arbitrary numpy operation on it and print the result on stdout.
EOF

runcmd "np -h"
runcmd "paste =(seq 10) =(seq 11 20) | np 'print(A)'"
runcmd "paste =(seq 10) =(seq 11 20) | np 'mu=A.mean(axis=0);print(mu)'"
runcmd "paste =(seq 10) =(seq 11 20) | np 'mu=A.mean(axis=1);print(mu)'"
echo "The python \`print\` command has been overwritten to print results as a shell friendly format. Therefore to print two variables you have to invoke the \'print\' command for each:"
runcmd "paste =(seq 10) =(seq 11 20) | np 'print(A.min());print(A.max())'"
echo "Change the delimiter with the -d option"
runcmd "paste -d ',' =(seq 10) =(seq 11 20) | np -d',' 'mu=A.mean(axis=0);print(mu)'"
