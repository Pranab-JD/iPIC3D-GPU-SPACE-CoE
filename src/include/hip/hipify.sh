#!/bin/bash

projectRoot="../"

# Define a mapping for input and output
declare -A fileMap=(
    ["${projectRoot}/cudaKernel"]="./hipKernel/"
    ["${projectRoot}/include/CUDA/"]="./include/"
    ["${projectRoot}/main/iPIC3Dlib.cu"]="./main/iPIC3Dlib.cpp"
)

hipifyScript="./hipify-perl"

# Iterate through the input-output mapping
for inputPath in "${!fileMap[@]}"; do
    outputPath="${fileMap[$inputPath]}"
    
    if [ -d "$inputPath" ]; then
        # Handle case where input is a directory
        echo "Processing directory: $inputPath"
        mkdir -p "$outputPath"
        # Process both .cu and .cuh files
        find "$inputPath" -type f \( -name "*.cu" -o -name "*.cuh" \) | while read -r file; do
            relativePath=$(realpath --relative-to="$inputPath" "$file")
            extension="${file##*.}"
            # Change output extension based on input file type
            if [ "$extension" = "cu" ]; then
                outputFile="$outputPath/${relativePath%.cu}.cpp"
            elif [ "$extension" = "cuh" ]; then
                outputFile="$outputPath/${relativePath%.cuh}.hpp"
            fi
            mkdir -p "$(dirname "$outputFile")"
            echo "Processing: $file -> $outputFile"
            perl "$hipifyScript" "$file" > "$outputFile"
        done
    elif [ -f "$inputPath" ]; then
        # Handle case where input is a file
        if [ -d "$outputPath" ]; then
            # Output is a directory
            mkdir -p "$outputPath"
            extension="${inputPath##*.}"
            # Change output extension based on input file type
            if [ "$extension" = "cu" ]; then
                outputFile="$outputPath/$(basename "${inputPath%.cu}.cpp")"
            elif [ "$extension" = "cuh" ]; then
                outputFile="$outputPath/$(basename "${inputPath%.cuh}.hpp")"
            fi
        else
            # Output is a specific file
            mkdir -p "$(dirname "$outputPath")"
            outputFile="$outputPath"
        fi
        echo "Processing: $inputPath -> $outputFile"
        perl "$hipifyScript" "$inputPath" > "$outputFile"
    else
        echo "Input path $inputPath does not exist, skipping..."
    fi
done

echo "Hipify-Perl done, post process start..."

./hipify-postProcess.sh

echo "Conversion completed."