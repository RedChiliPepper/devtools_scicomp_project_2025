#!/bin/bash

# URL of the Spambase dataset
URL="https://archive.ics.uci.edu/static/public/94/spambase.zip"
DEST_DIR="data"
ZIP_FILE="spambase.zip"

# Step 1: Download the dataset
echo "Downloading spambase.zip from $URL..."
curl -o $ZIP_FILE $URL

# Step 2: Create the 'data' directory if it doesn't exist
if [ ! -d "$DEST_DIR" ]; then
   echo "Creating directory: $DEST_DIR"
   mkdir $DEST_DIR
fi

# Step 3: Extract the dataset
echo "Extracting $ZIP_FILE..."
unzip $ZIP_FILE

# Step 4: Move the dataset to the 'data' folder
if [ -f "spambase.data" ]; then
   echo "Moving spambase.data to $DEST_DIR"
   mv spambase.data $DEST_DIR/
else
   echo "Error: spambase.data not found in the extracted files."
   exit 1
fi

# Step 5: Clean up additional files
echo "Cleaning up: Removing $ZIP_FILE and other unnecessary files..."
rm $ZIP_FILE
rm -f spambase.names spambase.DOCUMENTATION

echo "Download and extraction completed successfully."
