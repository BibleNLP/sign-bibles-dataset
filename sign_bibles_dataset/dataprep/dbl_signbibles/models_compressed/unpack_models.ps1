# PowerShell script to unpack compressed model files
# This script extracts all model files from the models_compressed directory to the sign-segmentation/models directory

# Ensure the target directory exists
$modelsDir = "../sign-segmentation/models"
if (-not (Test-Path $modelsDir)) {
    Write-Host "Creating models directory..."
    New-Item -Path $modelsDir -ItemType Directory -Force | Out-Null
}

# Extract the regular model files
Write-Host "Extracting regular model files..."
Expand-Archive -Path "groundingdino_swint_ogc.zip" -DestinationPath $modelsDir -Force
Expand-Archive -Path "yolov8_person-seg.zip" -DestinationPath $modelsDir -Force
Expand-Archive -Path "yolov8_hand_face-seg.zip" -DestinationPath $modelsDir -Force
Expand-Archive -Path "yolo11s-seg.zip" -DestinationPath $modelsDir -Force

# For the split SAM model, extract the parts and then combine them
Write-Host "Extracting and combining SAM model parts..."

# Create a temporary directory for the parts
$tempDir = "temp_sam_parts"
if (Test-Path $tempDir) {
    Remove-Item -Path $tempDir -Recurse -Force
}
New-Item -Path $tempDir -ItemType Directory -Force | Out-Null

# Extract each part
Expand-Archive -Path "sam_hq_vit_h.part1.zip" -DestinationPath $tempDir -Force
Expand-Archive -Path "sam_hq_vit_h.part2.zip" -DestinationPath $tempDir -Force
Expand-Archive -Path "sam_hq_vit_h.part3.zip" -DestinationPath $tempDir -Force

# Combine the parts into the original file
$outputFile = Join-Path $modelsDir "sam_hq_vit_h.pth"
$outputStream = [System.IO.File]::Create($outputFile)

for ($i = 1; $i -le 3; $i++) {
    $inputFile = Join-Path $tempDir "sam_hq_vit_h.pth.part$i"
    Write-Host "Adding part $i to combined file..."
    $inputStream = [System.IO.File]::OpenRead($inputFile)
    $inputStream.CopyTo($outputStream)
    $inputStream.Close()
}

$outputStream.Close()

# Clean up the temporary directory
Write-Host "Cleaning up temporary files..."
Remove-Item -Path $tempDir -Recurse -Force

Write-Host "All model files have been successfully extracted to $modelsDir"
