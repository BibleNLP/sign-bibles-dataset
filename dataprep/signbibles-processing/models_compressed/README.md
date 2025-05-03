# Compressed Model Files

This directory contains compressed versions of the model files used in the sign-segmentation component. The models have been compressed and, in the case of the large SAM model, split into parts to accommodate GitHub's file size limitations.

## Model Files

- `groundingdino_swint_ogc.zip` - Grounding DINO model for object detection
- `yolov8_person-seg.zip` - YOLOv8 model for person segmentation
- `yolov8_hand_face-seg.zip` - YOLOv8 model for hand and face segmentation
- `yolo11s-seg.zip` - YOLOv8 small model for segmentation
- `sam_hq_vit_h.part1.zip`, `sam_hq_vit_h.part2.zip`, `sam_hq_vit_h.part3.zip` - Split parts of the SAM HQ model

## How to Use

### Using the Unpacking Scripts

The easiest way to extract all model files is to use the provided scripts:

#### For Windows (PowerShell):
```
cd dataprep/signbibles-processing/models_compressed
./unpack_models.ps1
```

#### For Linux/Mac (Bash):
```
cd dataprep/signbibles-processing/models_compressed
chmod +x unpack_models.sh
./unpack_models.sh
```

These scripts will automatically extract all the compressed files and combine the split SAM model parts into a single file in the correct location.

### Manual Extraction

If you prefer to extract the files manually:

1. Create a `models` directory in the `sign-segmentation` directory if it doesn't exist:
   ```
   mkdir -p dataprep/signbibles-processing/sign-segmentation/models
   ```

2. Extract the regular model files:
   ```
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/groundingdino_swint_ogc.zip" -DestinationPath "dataprep/signbibles-processing/sign-segmentation/models/"
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/yolov8_person-seg.zip" -DestinationPath "dataprep/signbibles-processing/sign-segmentation/models/"
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/yolov8_hand_face-seg.zip" -DestinationPath "dataprep/signbibles-processing/sign-segmentation/models/"
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/yolo11s-seg.zip" -DestinationPath "dataprep/signbibles-processing/sign-segmentation/models/"
   ```

3. For the split SAM model, you need to extract the parts and then combine them:
   ```
   # Create a temporary directory for the parts
   mkdir -p temp_sam_parts
   
   # Extract each part
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/sam_hq_vit_h.part1.zip" -DestinationPath "temp_sam_parts/"
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/sam_hq_vit_h.part2.zip" -DestinationPath "temp_sam_parts/"
   Expand-Archive -Path "dataprep/signbibles-processing/models_compressed/sam_hq_vit_h.part3.zip" -DestinationPath "temp_sam_parts/"
   
   # Combine the parts into the original file
   $outputFile = "dataprep/signbibles-processing/sign-segmentation/models/sam_hq_vit_h.pth"
   $outputStream = [System.IO.File]::Create($outputFile)
   
   for ($i = 1; $i -le 3; $i++) {
       $inputFile = "temp_sam_parts/sam_hq_vit_h.pth.part$i"
       $inputStream = [System.IO.File]::OpenRead($inputFile)
       $inputStream.CopyTo($outputStream)
       $inputStream.Close()
   }
   
   $outputStream.Close()
   
   # Clean up the temporary directory
   Remove-Item -Path "temp_sam_parts" -Recurse -Force
   ```

## Alternative Download

If you prefer not to use these compressed files, you can also download the model files directly using the provided download scripts:

```
cd dataprep/signbibles-processing/sign-segmentation
python download_models.py
python download_dwpose.py
python download_grounded_sam.py
```

This will download the latest versions of the models from their respective sources.
