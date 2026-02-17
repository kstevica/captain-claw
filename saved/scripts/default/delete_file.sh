#!/bin/bash

# Script to delete a file
# Usage: ./delete_file.sh <filename>

FILE_TO_DELETE="test.txt"

if [ -z "$1" ]; then
    echo "No file specified. Using default: $FILE_TO_DELETE"
else
    FILE_TO_DELETE="$1"
fi

if [ -f "$FILE_TO_DELETE" ]; then
    rm "$FILE_TO_DELETE"
    echo "Deleted: $FILE_TO_DELETE"
else
    echo "File not found: $FILE_TO_DELETE"
fi
