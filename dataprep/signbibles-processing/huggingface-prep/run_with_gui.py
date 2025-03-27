#!/usr/bin/env python
"""
Run the sign language video processing pipeline with GUI progress display.
"""

import os
import sys
import argparse

def main():
    """Main function to run the processing pipeline with GUI."""
    parser = argparse.ArgumentParser(description="Process sign language videos with GUI progress display")
    parser.add_argument("--num-videos", type=int, default=1, help="Number of videos to download")
    parser.add_argument("--language", type=str, default="sqs", help="Language code to filter by")
    parser.add_argument("--project", type=str, help="Project name to filter by")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument("--clean", action="store_true", help="Clean previous outputs before processing")
    args = parser.parse_args()
    
    # Clean previous outputs if requested
    if args.clean:
        import shutil
        import os
        import re
        
        # Convert relative output directory to absolute path if needed
        output_dir = args.output_dir
        if not os.path.isabs(output_dir):
            # Get the absolute path relative to the project root
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            output_dir = os.path.join(project_root, output_dir)
        
        print(f"Cleaning output directory: {output_dir}")
        
        if os.path.exists(output_dir):
            # Clean segments directory - handle any language code, not just sqs
            language_code = args.language if args.language else "sqs"
            
            # First, try the specific language segments directory
            segments_dir = os.path.join(output_dir, "downloads", "downloads", language_code, "segments")
            if os.path.exists(segments_dir):
                print(f"Cleaning segments directory: {segments_dir}")
                # Remove all files in the segments directory
                for file in os.listdir(segments_dir):
                    file_path = os.path.join(segments_dir, file)
                    try:
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                            print(f"Removed: {file}")
                    except Exception as e:
                        print(f"Error removing {file}: {str(e)}")
            
            # Also check for any other language directories that might exist
            downloads_dir = os.path.join(output_dir, "downloads", "downloads")
            if os.path.exists(downloads_dir):
                for lang_dir in os.listdir(downloads_dir):
                    lang_segments_dir = os.path.join(downloads_dir, lang_dir, "segments")
                    if os.path.exists(lang_segments_dir) and lang_segments_dir != segments_dir:
                        print(f"Cleaning additional segments directory: {lang_segments_dir}")
                        for file in os.listdir(lang_segments_dir):
                            file_path = os.path.join(lang_segments_dir, file)
                            try:
                                if os.path.isfile(file_path):
                                    os.remove(file_path)
                                    print(f"Removed: {file}")
                            except Exception as e:
                                print(f"Error removing {file}: {str(e)}")
            
            # Clean processed directory
            processed_dir = os.path.join(output_dir, "processed")
            if os.path.exists(processed_dir):
                print(f"Cleaning processed directory: {processed_dir}")
                shutil.rmtree(processed_dir)
                os.makedirs(processed_dir, exist_ok=True)
            
            # Clean webdataset directory
            webdataset_dir = os.path.join(output_dir, "webdataset")
            if os.path.exists(webdataset_dir):
                print(f"Cleaning webdataset directory: {webdataset_dir}")
                shutil.rmtree(webdataset_dir)
                os.makedirs(webdataset_dir, exist_ok=True)
            
            # Remove manifest.json
            manifest_path = os.path.join(output_dir, "manifest.json")
            if os.path.exists(manifest_path):
                print(f"Removing manifest file: {manifest_path}")
                os.remove(manifest_path)
    
    # Import the main script
    from prepare_webdataset import main as prepare_main
    
    # Prepare arguments for the main script
    sys.argv = [
        sys.argv[0],
        "--num-videos", str(args.num_videos),
        "--output-dir", output_dir,
        "--with-gui"  # Always use GUI
    ]
    
    # Add language if specified
    if args.language:
        sys.argv.extend(["--language-code", args.language])
    
    # Add project if specified
    if args.project:
        sys.argv.extend(["--project-name", args.project])
    
    # Print the command that will be run
    print(f"Running command: {' '.join(sys.argv)}")
    
    # Run the main function
    prepare_main()

if __name__ == "__main__":
    main()
