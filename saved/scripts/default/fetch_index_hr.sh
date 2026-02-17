#!/bin/bash

# Script to fetch text content from www.index.hr and save it to testfetch.txt

URL="https://www.index.hr"
OUTPUT_FILE="testfetch.txt"

echo "Fetching content from $URL..."

# Fetch the content and save to file
curl -s -L "$URL" -o "$OUTPUT_FILE"

if [ $? -eq 0 ]; then
    echo "Content successfully saved to $OUTPUT_FILE"
    echo "File size: $(wc -c < "$OUTPUT_FILE") bytes"
else
    echo "Error: Failed to fetch content from $URL"
    exit 1
fi
