#!/usr/bin/env bash
# shellcheck shell=bash
# -*- coding: UTF8 -*-

grep -E "^.+[[:space:]]*\(\)[[:space:]]*{" $1 | grep -v "^#"
