#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Mon May 19 11:12:03 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Full path to the directory of the current script
DIRSCRIPT="$(dirname "$(readlink -f "$0")")"
# MYTMP=$(mktemp -d)  # Temporary directory for the current script. Use it to put temporary files.
# trap 'rm -rvf "$MYTMP"' EXIT INT  # Will be removed at the end of the script

function usage () {
    cat << EOF
Usage: $(basename "$0") [OPTIONS] [SEARCH TERMS]
Search the web using a specified search engine.

Options:
    -g, --google   Use Google as the search engine (default).
    -w, --wikipedia Use Wikipedia as the search engine.
    -l, --location LOCATION  Specify the location for the search (default: 'en'), only for Wikipedia.
    -h, --help print this help message and exit
EOF
}

# Default search engine
SEARCH_ENGINE="google"
LOCATION="en"
# Parse command line options
while [[ $# -gt 0 ]]; do
    case "$1" in
        -g|--google)
            SEARCH_ENGINE="google"
            shift
            ;;
        -w|--wikipedia)
            SEARCH_ENGINE="wikipedia"
            shift
            ;;
        -l|--location)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                LOCATION="$2"
                shift 2
            else
                echo "Error: --location requires an argument."
                exit 1
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Check if search terms are provided

if [[ $# -eq 0 ]]; then
    echo "Error: No search terms provided."
    usage
    exit 1
fi
# Join the search terms into a single string
SEARCH_TERMS="$*"
echo "Searching for: $SEARCH_TERMS"
case "$SEARCH_ENGINE" in
    google)
        # URL encode the search terms
        SEARCH_TERMS_ENCODED=$(echo "$SEARCH_TERMS" | jq -sRr @uri)
        # Open Google search in the default web browser
        xdg-open "https://www.google.com/search?q=$SEARCH_TERMS_ENCODED" &>/dev/null &
        ;;
    wikipedia)
        # URL encode the search terms
        SEARCH_TERMS_ENCODED=$(echo "$SEARCH_TERMS" | jq -sRr @uri)
        # Open Wikipedia search in the default web browser
        xdg-open "https://$LOCATION.wikipedia.org/w/index.php?search=$SEARCH_TERMS_ENCODED" &>/dev/null &
        ;;
    *)
        echo "Error: Unsupported search engine '$SEARCH_ENGINE'."
        exit 1
        ;;
esac
