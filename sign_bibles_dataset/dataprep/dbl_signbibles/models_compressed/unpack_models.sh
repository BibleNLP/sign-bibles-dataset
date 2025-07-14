#!/bin/bash
# Bash script to unpack compressed model files
# This script extracts all model files from the models_compressed directory to the sign-segmentation/models directory

# Ensure the target directory exists
MODELS_DIR="../sign-segmentation/models"
mkdir -p "$MODELS_DIR"

# Extract the regular model files
echo "Extracting regular model files..."
unzip -o "groundingdino_swint_ogc.zip" -d "$MODELS_DIR"
unzip -o "yolov8_person-seg.zip" -d "$MODELS_DIR"
unzip -o "yolov8_hand_face-seg.zip" -d "$MODELS_DIR"
unzip -o "yolo11s-seg.zip" -d "$MODELS_DIR"

# For the split SAM model, extract the parts and then combine them
echo "Extracting and combining SAM model parts..."

# Create a temporary directory for the parts
TEMP_DIR="temp_sam_parts"
rm -rf "$TEMP_DIR"
mkdir -p "$TEMP_DIR"

# Extract each part
unzip -o "sam_hq_vit_h.part1.zip" -d "$TEMP_DIR"
unzip -o "sam_hq_vit_h.part2.zip" -d "$TEMP_DIR"
unzip -o "sam_hq_vit_h.part3.zip" -d "$TEMP_DIR"

# Combine the parts into the original file
OUTPUT_FILE="$MODELS_DIR/sam_hq_vit_h.pth"
rm -f "$OUTPUT_FILE"
touch "$OUTPUT_FILE"

for i in {1..3}; do
    echo "Adding part $i to combined file..."
    cat "$TEMP_DIR/sam_hq_vit_h.pth.part$i" >> "$OUTPUT_FILE"
done

# Clean up the temporary directory
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"

echo "All model files have been successfully extracted to $MODELS_DIR"
