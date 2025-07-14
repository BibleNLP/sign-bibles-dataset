import json
import time
from pathlib import Path

import cv2
import langcodes
import requests

from sign_bibles_dataset.dataprep.dbl_signbibles.dbl_sign import dbl_manifest_generator

PARENT_DIR = Path(__file__).resolve().parent.parent
DBL_SIGN_DIR = PARENT_DIR / "dbl_sign"


class DBLSignDownloader:
    """Class to handle downloading videos from DBL-sign."""

    def __init__(self, output_dir: Path | str):
        """Initialize the downloader with output directory."""
        self.output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        self.manifest_path = DBL_SIGN_DIR / "manifest.json"
        self.manifest = None

        # Load manifest
        self.load_manifest()

    def generate_fresh_manifest(self):
        """Generate a fresh manifest using the imported manifest generator functions."""
        print("Generating fresh manifest...")

        # Response data containing the list of sign language projects (hardcoded in the original script)
        response_data = dbl_manifest_generator.get_response_data()

        # Create manifest using the imported function
        manifest = dbl_manifest_generator.create_manifest(response_data)

        # Save manifest using the imported function
        dbl_manifest_generator.save_manifest(manifest, self.manifest_path)

        print("Fresh manifest generated successfully.")

        # Load the newly generated manifest
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        return self.manifest

    def load_manifest(self):
        """Load the manifest file."""
        if not self.manifest_path.exists():
            print("Manifest file not found. Generating a fresh one...")
            return self.generate_fresh_manifest()

        # Check if manifest is older than 24 hours
        manifest_age = time.time() - self.manifest_path.stat().st_mtime
        if manifest_age > 86400:  # 24 hours in seconds
            print("Manifest is older than 24 hours. Generating a fresh one...")
            return self.generate_fresh_manifest()

        print("Loading existing manifest...")
        with open(self.manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)

        return self.manifest

    def refresh_manifest(self):
        """Refresh the manifest using the manifest generator script."""
        try:
            self.generate_fresh_manifest()
        except Exception as e:
            raise Exception(f"Failed to refresh manifest: {e}") from None

    def download_videos(self, num_videos, language_codes=None, project_name=None):
        """
        Download a specified number of videos from DBL-sign.

        Args:
            num_videos: Number of videos to download
            language_codes: Optional filter by language codes
            project_name: Optional filter by project name

        Returns:
            List of dictionaries with video information

        """
        if "languages" not in self.manifest:
            raise ValueError("Invalid manifest structure: 'languages' key not found")

        if language_codes is None:
            language_codes = [language_codes]
        all_downloaded_videos = []
        for language_code in language_codes:
            # Filter projects based on criteria
            filtered_projects = []
            total_languages = len(self.manifest["languages"])
            print(f"Total Languages in Manifest: {total_languages}")
            for i, (lang_code, projects) in enumerate(self.manifest["languages"].items()):
                if language_code and lang_code != language_code:
                    continue

                for proj_name, proj_info in projects.items():
                    # Only filter by project name if it's explicitly provided
                    if project_name and proj_name != project_name:
                        continue

                    # Count MP4 files in this project
                    mp4_count = sum(
                        1 for file_info in proj_info["files"] if file_info["filename"].lower().endswith(".mp4")
                    )

                    if mp4_count > 0:
                        filtered_projects.append((lang_code, proj_name, proj_info))

            if not filtered_projects:
                filter_criteria = f"language_code={language_code}"
                if project_name:
                    filter_criteria += f", project_name={project_name}"
                error_msg = f"No projects found matching the criteria ({filter_criteria})"

                raise ValueError(error_msg)

            # Download videos
            downloaded_videos = []
            videos_downloaded = 0

            print(f"Found {len(filtered_projects)} projects matching the criteria")

            for i, (lang_code, proj_name, proj_info) in enumerate(filtered_projects):
                if videos_downloaded >= num_videos:
                    break

                project_dir = self.output_dir / lang_code / proj_name
                print(f"Project Dir: {project_dir}")
                project_dir.mkdir(exist_ok=True, parents=True)

                # Get metadata for this project
                metadata = {
                    "language": {
                        "ISO639-3": lang_code,
                        "BCP-47": langcodes.standardize_tag(lang_code),
                    },
                    "project_name": proj_name,
                    "copyright": proj_info.get("rights_holder", ""),
                    "license": proj_info.get("license", ""),
                    "source": proj_info.get("url", ""),
                }

                # Save metadata
                with open(project_dir / "metadata.json", "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)

                # Download MP4 files
                mp4_files = [f for f in proj_info["files"] if f["filename"].lower().endswith(".mp4")]
                for j, file_info in enumerate(mp4_files):
                    if videos_downloaded >= num_videos:
                        break

                    filename = file_info["filename"]
                    filepath = Path(project_dir) / filename
                    url = file_info["download_url"]

                    # Skip if file already exists
                    if filepath.exists() and filepath.stat().st_size > 0:
                        # print(f"File already exists: {filepath}")

                        downloaded_videos.append({"path": str(filepath), **metadata})
                        videos_downloaded += 1
                        continue

                    # Download the file
                    print(f"Downloading {url} to {filepath}")

                    try:
                        response = requests.get(url, stream=True)
                        response.raise_for_status()

                        # Show progress bar
                        downloaded_size = 0
                        block_size = 8192

                        with open(filepath, "wb") as f:
                            for chunk in response.iter_content(chunk_size=block_size):
                                if chunk:
                                    f.write(chunk)
                                    downloaded_size += len(chunk)

                        # Validate MP4

                        try:
                            cap = cv2.VideoCapture(filepath)
                            if not cap.isOpened():
                                error_msg = f"Warning: Could not open downloaded file as video: {filepath}"
                                print(error_msg)

                                continue
                            cap.release()

                        except Exception as e:
                            error_msg = f"Error validating video: {e}"
                            print(error_msg)

                            continue

                        downloaded_videos.append({"path": filepath, **metadata})
                        videos_downloaded += 1

                    except Exception as e:
                        error_msg = f"Error downloading {url}: {e}"
                        print(error_msg)

            print(f"Downloaded {videos_downloaded} videos")
            all_downloaded_videos.extend(downloaded_videos)
        return all_downloaded_videos
