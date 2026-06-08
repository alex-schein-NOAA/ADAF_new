#!/bin/bash

# Define your directories
DIRS=("/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/train_data" "/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/valid_data" "/scratch5/BMC/ai-datadepot/projects/aschein/ADAF_new/data/test_data")
OUTPUT_FILE="missing_files.txt"

# Initialize (or clear out) the output file at the start of the script
> "$OUTPUT_FILE"

echo "Checking for missing files..."
echo "Detailed missing list will be saved to: $OUTPUT_FILE"
echo "--------------------------------------------------------"

total_missing_all_dirs=0

for dir in "${DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "⚠️ Directory '$dir' does not exist. Skipping."
        echo "--------------------------------------------------------"
        continue
    fi

    echo "Scanning directory: $dir"

    # Get a sorted list of all matching files in the directory
    files=($(find "$dir" -maxdepth 1 -name "????-??-??_??.nc" -exec basename {} \; | sort))

    # If the directory has no matching files, move to the next one
    if [ ${#files[@]} -eq 0 ]; then
        echo "ℹ️ No matching NetCDF files found in this directory."
        echo "--------------------------------------------------------"
        continue
    fi

    # Identify the first and last files
    first_file="${files[0]}"
    last_file="${files[${#files[@]}-1]}"

    # Convert filenames into standard date strings
    first_temp="${first_file%.nc}"
    first_date="${first_temp/_/ }:00:00"

    last_temp="${last_file%.nc}"
    last_date="${last_temp/_/ }:00:00"

    # Convert date strings to UTC epoch timestamps
    current_sec=$(date -u -d "$first_date" +%s)
    end_sec=$(date -u -d "$last_date" +%s)

    missing_count=0

    # Loop hour by hour from the start file to the end file
    while [ "$current_sec" -le "$end_sec" ]; do
        filename=$(date -u -d "@$current_sec" +"%Y-%m-%d_%H.nc")

        # Check if the generated filename exists in the directory
        if [ ! -f "$dir/$filename" ]; then
            # Append the missing file path to the output text file
            echo "$dir/$filename" >> "$OUTPUT_FILE"
            ((missing_count++))
        fi

        # Advance by 1 hour (3600 seconds)
        current_sec=$((current_sec + 3600))
    done

    if [ "$missing_count" -eq 0 ]; then
        echo " Scan complete! No missing hours found between endpoints."
    else
        echo " Found $missing_count missing files (logged to text file)."
        total_missing_all_dirs=$((total_missing_all_dirs + missing_count))
    fi
    echo "--------------------------------------------------------"
done

echo "Process finished!"
echo "Total missing files across all directories: $total_missing_all_dirs"
echo "List of missing files saved to $OUTPUT_FILE"
