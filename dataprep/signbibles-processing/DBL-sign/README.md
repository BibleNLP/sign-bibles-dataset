# DBL Sign Language Video Downloader

A Python-based tool for downloading and managing sign language video content from the Digital Bible Library (DBL). Features include:
- Modern GUI with progress tracking
- Automatic manifest management
- Optional S3 upload support
- MP4 validation
- Download resumption capability

## Requirements

- Python 3.6+
- Required packages (install via pip):
  ```
  requests
  boto3
  beautifulsoup4
  ```

## Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/sil-ai/dbl-sign.git
   cd dbl-sign
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the downloader:
   ```bash
   python DBL-manifest-downloader.py
   ```

## Usage

### Basic Download
```bash
python DBL-manifest-downloader.py
```

### With File Limit
Limit the number of MP4 files to download:
```bash
python DBL-manifest-downloader.py --limit 200
```

### With S3 Upload
Upload files to an S3 bucket as they're downloaded:
```bash
python DBL-manifest-downloader.py --s3-bucket your-bucket-name
```

### With Authentication
If the manifest requires authentication:
```bash
python DBL-manifest-downloader.py --auth username:password
```

### Full Options
```bash
python DBL-manifest-downloader.py --help
```

## Features

### Modern GUI
- Real-time progress tracking for both overall progress and current file
- Clear status messages
- Progress bars for download and upload operations

### Manifest Management
- Automatic manifest refresh if older than 3 hours
- Tracks download progress for resume capability
- Validates MP4 files before marking as complete

### S3 Integration
- Optional upload to S3 bucket
- Progress tracking for uploads
- Maintains folder structure in S3

### Error Handling
- Graceful handling of network issues
- Download resumption after interruption
- Invalid file detection and reporting

## Project Structure

- `DBL-manifest-downloader.py`: Main script for downloading files
- `dbl_utils.py`: Helper classes for GUI, logging, and S3 operations
- `DBL-manifest-generator.py`: Support script for manifest generation (automatically called when needed)

## Notes

- The downloader will create a `downloads` directory for local files
- Download progress is saved and can be resumed if interrupted
- MP4 files are validated before being marked as complete
- The manifest is automatically refreshed if it's older than 3 hours


