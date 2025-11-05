#!/usr/bin/env bash

#############################################################################
# Author: Guillaume Bouvier -- guillaume.bouvier@pasteur.fr                 #
# https://research.pasteur.fr/en/member/guillaume-bouvier/                  #
# Copyright (c) 2025 Institut Pasteur                                       #
#############################################################################
#
# creation_date: Fri Oct 24 13:56:06 2025

set -e  # exit on error
set -o pipefail  # exit when a process in the pipe fails
set -o noclobber  # prevent overwriting redirection

# Help message function
function usage {
    echo "Usage: $0 [-s FIELD_SEPARATOR] <input_file>"
    echo "  -s FIELD_SEPARATOR: Set the field separator (default: ',')"
    exit 1
}

FIELD_SEPARATOR="," # Default field separator

# Parse command line options
while getopts "s:h" OPTION; do
    case "$OPTION" in
        s)
            FIELD_SEPARATOR="$OPTARG"
            ;;
        h)
            usage
            ;;
        *)
            usage
            ;;
    esac
done
shift "$((OPTIND - 1))" # Shift arguments so that $1 refers to the first non-option argument

# Check if an input file is provided
if [ -z "$1" ]; then
    usage
fi

if grep -q $'\r' "$1"; then
  dos2unix "$1"
fi

awk -v FS="$FIELD_SEPARATOR" '
  BEGIN {
    # Define the Field Separator (FS) for CSV (default: comma).
    # This is set externally via -v FS="..."
    
    # Initialize a variable to track if we are currently inside a multi-line field
    # 0 = not inside, 1 = inside
    in_multiline_field = 0
  }
  
  {
    # 1. Count the double quotes on the current line.
    # The gsub function returns the number of replacements made, 
    # but using a substitution that replaces a character with itself (like /,/,"&")
    # is a common awk trick to just count occurrences.
    quotes = gsub(/"/, "&", $0) 
    
    # 2. Update the state: Check if the quote count is odd.
    if (quotes % 2 != 0) {
      in_multiline_field = !in_multiline_field
    }
    
    # 3. Processing logic based on the state:
    if (in_multiline_field) {
      # We are inside a multi-line field (or have just started one).
      # Join the current line with the next one.
      # The newline is replaced by a space (or remove the space for a tight join).
      printf "%s ", $0 
    } else {
      # We are NOT inside a multi-line field (or have just finished one).
      # Print the previously buffered content (if any) and the current line.
      # This line marks the end of a complete record.
      print $0
    }
  }
' "$1"
