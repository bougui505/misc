#!/usr/bin/env zsh
# -*- coding: UTF8 -*-

cc -fPIC -shared -o lib/initialize_matrix.so initialize_matrix.c
cc -fPIC -shared -o lib/smith_waterman.so smith_waterman.c