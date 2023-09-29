#!/usr/bin/env bash
# shellcheck shell=bash
# -*- coding: UTF8 -*-

grep -E '^[[:space:]]*([[:alnum:]_]+[[:space:]]*\(\)|function[[:space:]]+[[:alnum:]_]+)' $1

