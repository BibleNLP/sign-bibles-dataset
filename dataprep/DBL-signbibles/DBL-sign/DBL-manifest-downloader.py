import json
import os
import requests
from pathlib import Path
import argparse
import threading
import time
from dbl_utils import DownloadLog, DownloadProgressWindow, S3Storage, validate_mp4

def refresh_manifest(manifest_path, auth=None):
    """Refresh the manifest using the manifest generator script."""
    try:
        import subprocess
        cmd = ['python', 'DBL-manifest-generator.py']
        if auth:
            cmd.extend(['--username', auth[0], '--password', auth[1]])
        subprocess.run(cmd, check=True)
        
        with open(manifest_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"Failed to refresh manifest: {e}")

def check_manifest_timestamp(manifest):
    """Check if manifest needs to be refreshed (older than 3 hours)."""
    if "timestamp" not in manifest:
        return True
    
    current_time = int(time.time())
    manifest_age = current_time - manifest["timestamp"]
    return manifest_age > (3 * 60 * 60)  # 3 hours in seconds

def download_file(url, filepath, session, progress_window, s3_storage=None):
    """Downloads a file from the URL to the specified filepath or S3."""
    temp_filepath = filepath.with_suffix('.tmp')
    
    try:
        # Start the download
        progress_window.update_status(f"Downloading: {filepath}")
        progress_window.update_file_progress(0, f"Downloading: {filepath}")
        
        response = session.get(url, stream=True)
        response.raise_for_status()
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        
        # Download to temporary file first
        with open(temp_filepath, 'wb') as f:
            if file_size == 0:
                f.write(response.content)
                progress_window.update_file_progress(100, f"Downloaded: {filepath}")
            else:
                downloaded = 0
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        progress = (downloaded / file_size) * 100
                        progress_window.update_file_progress(progress, f"Downloading: {filepath}")
        
        # For S3 storage
        if s3_storage:
            # For MP4 files, validate before uploading
            if filepath.suffix.lower() == '.mp4':
                progress_window.update_status(f"Validating MP4: {filepath}")
                progress_window.update_file_progress(0, f"Validating: {filepath}")
                if not validate_mp4(temp_filepath):
                    progress_window.update_status(f"Invalid MP4 file: {filepath}")
                    if temp_filepath.exists():
                        temp_filepath.unlink()
                    return False
            
            # Read the file and upload to S3
            try:
                progress_window.update_status(f"Uploading to S3: {filepath}")
                progress_window.update_file_progress(0, f"Uploading to S3: {filepath}")
                
                with open(temp_filepath, 'rb') as f:
                    file_data = f.read()
                success = s3_storage.upload_file(file_data, filepath, progress_window)
                
                if success:
                    progress_window.update_status(f"Successfully uploaded: {filepath}")
                    progress_window.update_file_progress(100, f"Uploaded: {filepath}")
                else:
                    progress_window.update_status(f"Failed to upload: {filepath}")
            finally:
                # Always clean up temp file
                if temp_filepath.exists():
                    temp_filepath.unlink()
            return success
        else:
            if filepath.exists():
                if not filepath.suffix.lower() == '.mp4' or validate_mp4(filepath):
                    progress_window.update_status(f"File already exists and is valid: {filepath}")
                    temp_filepath.unlink()
                    return True
            
            # Move temp file to final location
            if temp_filepath.exists():
                temp_filepath.replace(filepath)
                progress_window.update_file_progress(100, f"Downloaded: {filepath}")
                progress_window.update_status(f"Successfully downloaded: {filepath}")
                return True
            return False
            
    except requests.exceptions.RequestException as e:
        progress_window.update_status(f"Error downloading {url}: {e}")
        progress_window.update_file_progress(0, "Error")
        if temp_filepath.exists():
            temp_filepath.unlink()
        return False
    except Exception as e:
        progress_window.update_status(f"Unexpected error: {e}")
        progress_window.update_file_progress(0, "Error")
        if temp_filepath.exists():
            temp_filepath.unlink()
        return False

def download_manifest_files(manifest_path, base_dir="downloads", auth=None, file_limit=None, s3_storage=None):   
    # Create progress window
    progress_window = DownloadProgressWindow()
    base_dir = Path(base_dir)
    
    def download_thread():
        try:
            # Initialize download log
            log_file = Path(manifest_path).parent / f"{Path(manifest_path).stem}_download.log"
            download_log = DownloadLog(log_file)
            
            progress_window.update_status("Loading manifest file...")
            
            # Load manifest
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
            except FileNotFoundError:
                raise Exception(f"Manifest file not found: {manifest_path}")
            except json.JSONDecodeError as e:
                raise Exception(f"Invalid manifest JSON: {e}")
            
            # Check if manifest needs refresh
            if check_manifest_timestamp(manifest):
                progress_window.update_status("Manifest is older than 3 hours. Refreshing...")
                try:
                    manifest = refresh_manifest(manifest_path, auth)
                except Exception as e:
                    raise Exception(f"Failed to refresh manifest: {e}")
                progress_window.update_status("Manifest refreshed successfully.")
            
            # Verify manifest structure
            if "languages" not in manifest:
                raise Exception("Invalid manifest structure: 'languages' key not found")
            
            total_mp4_files = 0  # Only count MP4 files
            mp4_count = 0
            existing_files = []
            files_to_download = []
            projects_to_process = {}  # Keep track of which projects to fully process
            
            # Create a session for all requests
            session = requests.Session()
            if auth:
                session.auth = auth
            
            # First pass: check which files exist and analyze MP4 counts
            progress_window.update_status("Checking existing files...")
            
            try:
                # First count total MP4s to determine which projects to include
                all_projects = []
                for lang_code, projects in manifest["languages"].items():
                    for project_name, project_info in projects.items():
                        project_mp4_count = sum(1 for file_info in project_info["files"] 
                                             if file_info['filename'].lower().endswith('.mp4'))
                        if file_limit is None or mp4_count + project_mp4_count <= file_limit:
                            mp4_count += project_mp4_count
                            all_projects.append((lang_code, project_name, project_info))
                        if file_limit is not None and mp4_count >= file_limit:
                            break
                    if file_limit is not None and mp4_count >= file_limit:
                        break
                
                progress_window.update_status(f"Found {len(all_projects)} projects to process...")
                
                # Now process files for selected projects
                for lang_code, project_name, project_info in all_projects:
                    project_dir = base_dir / lang_code / project_name
                    
                    for file_info in project_info["files"]:
                        filename = file_info['filename']
                        filepath = project_dir / filename
                        is_mp4 = filename.lower().endswith('.mp4')
                        
                        if is_mp4:
                            total_mp4_files += 1
                        
                        if s3_storage:
                            exists = s3_storage.file_exists(filepath)
                        else:
                            exists = ((filepath.exists() and 
                                    (not is_mp4 or validate_mp4(filepath))) or 
                                    download_log.is_completed(filepath))
                        
                        if exists:
                            existing_files.append((filepath, file_info, is_mp4))
                        else:
                            files_to_download.append((filepath, file_info, is_mp4))
                    
                    projects_to_process[(lang_code, project_name)] = project_info
                
                if not all_projects:
                    raise Exception("No projects found to process")
                
                # Process license info for selected projects
                for (lang_code, project_name), project_info in projects_to_process.items():
                    project_dir = base_dir / lang_code / project_name
                    if not s3_storage:
                        project_dir.mkdir(parents=True, exist_ok=True)
                        license_file = project_dir / "license_info.txt"
                        with open(license_file, 'w', encoding='utf-8') as f:
                            f.write(f"License: {project_info.get('license', 'Unknown')}\n")
                            f.write(f"Rights Holder: {project_info.get('rights_holder', 'Unknown')}\n")
                            f.write(f"Project URL: {project_info.get('url', 'Unknown')}\n")
                    else:
                        license_info = (
                            f"License: {project_info.get('license', 'Unknown')}\n"
                            f"Rights Holder: {project_info.get('rights_holder', 'Unknown')}\n"
                            f"Project URL: {project_info.get('url', 'Unknown')}\n"
                        )
                        license_path = project_dir / "license_info.txt"
                        s3_storage.upload_file(license_info.encode('utf-8'), license_path)
                
                # Update total count and initial progress
                current_mp4_count = sum(1 for _, _, is_mp4 in existing_files if is_mp4)
                overall_progress = (current_mp4_count * 100) / (file_limit or total_mp4_files)
                
                progress_window.update_status(
                    f"Found {len(existing_files)} existing files ({current_mp4_count} MP4s) and {len(files_to_download)} files to download "
                    f"({sum(1 for _, _, is_mp4 in files_to_download if is_mp4)} MP4s)"
                )
                progress_window.update_overall_progress(overall_progress, current_mp4_count, file_limit or total_mp4_files)
                
                # Second pass: download missing files
                for filepath, file_info, is_mp4 in files_to_download:
                    progress_window.update_status(f"Processing: {filepath}")
                    progress_window.update_file_progress(0, f"Downloading: {filepath}")
                    
                    success = download_file(file_info['download_url'], filepath, session, progress_window, s3_storage)
                    
                    if success:
                        download_log.mark_completed(filepath)
                        existing_files.append((filepath, file_info, is_mp4))
                        if is_mp4:
                            current_mp4_count += 1
                            overall_progress = (current_mp4_count * 100) / (file_limit or total_mp4_files)
                            progress_window.update_overall_progress(overall_progress, current_mp4_count, file_limit or total_mp4_files)
                    else:
                        progress_window.update_status(f"Failed to download {filepath}")
                
                # If all files were downloaded successfully, clean up the log file
                final_mp4_count = sum(1 for _, _, is_mp4 in existing_files if is_mp4)
                if final_mp4_count == total_mp4_files:
                    download_log.cleanup()
                    progress_window.update_status("Download complete! Log file cleaned up.")
                else:
                    progress_window.update_status(
                        f"Download complete with some failures. {final_mp4_count} of {total_mp4_files} MP4 files processed. "
                        "Log file preserved for resume capability."
                    )
                
            except Exception as e:
                raise Exception(f"Error during file processing: {e}")
            
            progress_window.update_status("Processing complete.")
            progress_window.complete()
            
        except Exception as e:
            error_msg = f"An error occurred: {str(e)}"
            print(error_msg)  # Print to console as well
            progress_window.update_status(error_msg)
            progress_window.complete()
    
    # Start download thread
    thread = threading.Thread(target=download_thread)
    thread.daemon = True
    thread.start()
    
    # Start the GUI event loop
    progress_window.root.mainloop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download files from a manifest JSON file.')
    parser.add_argument('--manifest', default='manifest.json', help='Path to the manifest JSON file')
    parser.add_argument('--output', default='downloads', help='Base directory for downloaded files')
    parser.add_argument('--username', help='Username for authentication')
    parser.add_argument('--password', help='Password for authentication')
    parser.add_argument('--limit', type=int, help='Limit the number of MP4 files to download')
    parser.add_argument('--s3-bucket', help='S3 bucket to upload files to')
    parser.add_argument('--s3-folder', default='unprocessed', help='Folder in S3 bucket to upload files to')
    
    args = parser.parse_args()
    
    auth = (args.username, args.password) if args.username and args.password else None
    s3_storage = S3Storage(args.s3_bucket, args.s3_folder) if args.s3_bucket else None
    
    download_manifest_files(args.manifest, args.output, auth, args.limit, s3_storage)
