#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

[[ ! -d lib ]] && mkdir lib
gcc -fPIC -shared -o lib/initialize_matrix.so initialize_matrix.c
gcc -fPIC -shared -o lib/smith_waterman.so smith_waterman.c
