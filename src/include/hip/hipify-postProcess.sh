#!/bin/bash

# Define the root directory for the operation
rootDir="."

# Define the replacement list (case-sensitive)
declare -A replacements=(
    ["cuda"]="hip"   
    ["CUDA"]="HIP"   
    ["cuh"]="hpp"
    ["CUH"]="HPP"
)

# Find all .cpp and .hpp files recursively
find "$rootDir" -type f \( -name "*.cpp" -o -name "*.hpp" \) | while read -r file; do
    # Process file content
    echo "Processing file content: $file"
    tempFile="${file}.tmp"
    cp "$file" "$tempFile"
    for key in "${!replacements[@]}"; do
        sed -i "s/$key/${replacements[$key]}/g" "$tempFile"
    done
    mv "$tempFile" "$file"
    
    # Process file name
    newFileName="$file"
    for key in "${!replacements[@]}"; do
        newFileName=$(echo "$newFileName" | sed "s/$key/${replacements[$key]}/g")
    done
    if [ "$newFileName" != "$file" ]; then
        echo "Renaming: $file -> $newFileName"
        mv "$file" "$newFileName"
    fi
done

echo "Text and file name conversion completed."