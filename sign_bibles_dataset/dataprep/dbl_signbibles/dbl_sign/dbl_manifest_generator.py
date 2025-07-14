import argparse
import json
import re
import time
import urllib.parse
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# Originally based on this: https://colab.research.google.com/drive/1_uFpG9lKJHV2S_jAUB5OA_W3m30BT14b?usp=sharing


def create_manifest(response_data, auth=None):
    """Creates a manifest of files organized by language code and project name."""
    manifest = {
        "timestamp": int(time.time()),  # Add timestamp when manifest is created
        "languages": {},
    }

    # Create a session for all requests
    session = requests.Session()
    if auth:
        session.auth = auth

    # Create progress bar for entries
    entries = response_data["aaData"]
    pbar = tqdm(entries, desc="Processing entries", unit="entry", position=0)

    for entry in pbar:
        id_val, _, lang_code, _, project_name, rights_holder = entry
        pbar.set_postfix_str(f"Project: {project_name[:30]}...")

        # Initialize nested structure if not exists
        if lang_code not in manifest["languages"]:
            manifest["languages"][lang_code] = {}
        if project_name not in manifest["languages"][lang_code]:
            manifest["languages"][lang_code][project_name] = {
                "files": [],
                "license": None,
                "rights_holder": rights_holder,
                "url": f"https://app.thedigitalbiblelibrary.org/entry?id={id_val}",
            }

        # Get the license text and download URL from the entry page
        time.sleep(2)  # Wait between requests
        license_text, download_url = get_entry_info(id_val, session)

        if license_text:
            manifest["languages"][lang_code][project_name]["license"] = license_text

        if download_url:
            # Get the download listing
            time.sleep(2)  # Wait between requests
            url = f"https://app.thedigitalbiblelibrary.org{download_url}"

            response = session.get(url)
            if response.status_code == 200:
                files = extract_files_from_html(response.text)
                manifest["languages"][lang_code][project_name]["files"].extend(files)
            else:
                pbar.write(f"Error fetching download listing: Status code {response.status_code}")
        else:
            pbar.write(f"No download URL found for {project_name}")

    return manifest


def get_entry_info(id_val, session):
    """Fetches both the license text and download URL from the entry page."""
    try:
        url = f"https://app.thedigitalbiblelibrary.org/entry?id={id_val}"
        response = session.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.text, "html.parser")
            license_text = None
            download_url = None

            # Look for license text
            for text in soup.stripped_strings:
                if "(Creative Commons" in text and ")" in text:
                    start = text.find("(")
                    end = text.find(")", start)
                    if start != -1 and end != -1:
                        license_text = text[start + 1 : end]
                        break

            # Look for download link
            download_link = soup.find("a", href=lambda href: href and "download_listing" in href)
            if download_link:
                download_url = download_link["href"]

            return license_text, download_url

        return None, None
    except Exception as e:
        print(f"Error fetching entry info for ID {id_val}: {e}")
        return None, None


def extract_files_from_html(html_content):
    """Extracts file information from the HTML content."""
    files = []
    soup = BeautifulSoup(html_content, "html.parser")

    def process_node(node):
        """Recursively processes a node to extract file information."""
        # Create progress bar for file extraction, but don't keep it after completion
        items = node.find_all("li", recursive=False)
        for li in tqdm(items, desc="Extracting files", leave=False, position=1):
            if li.find("input", type="checkbox"):
                # Process nested folders
                nested_ul = li.find("ul")
                if nested_ul:
                    process_node(nested_ul)
            elif li.find("a"):  # File case
                download_link = li.find("a")["href"]

                # Extract filename from the download link
                if "s3.amazonaws.com" in download_link:
                    match = re.search(r"filename%3D([^&]+)", download_link)
                    if match:
                        filename = urllib.parse.unquote(match.group(1))
                        filename = Path(filename).name
                    else:
                        filename = Path(download_link).name
                else:
                    filename = Path(download_link).name

                files.append({"filename": filename, "download_url": download_link})

    treeview_div = soup.find("div", class_="treeview")
    if treeview_div:
        ul_element = treeview_div.find("ul")
        if ul_element:
            process_node(ul_element)

    return files


def save_manifest(manifest, output_file="manifest.json"):
    """Saves the manifest to a JSON file."""
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(f"Manifest saved to {output_file}")


def get_response_data() -> dict:
    response_data = {
        "aaData": [
            [
                "245cca6ac5984130",
                "Sri Lankan Sign Language",
                "sqs",
                "Sri Lanka",
                "Chronological Bible Translation in Sri Lankan Sign Language",
                "DOOR International",
            ],
            [
                "055195093e2347d0",
                "Burundian Sign Language",
                "lsb",
                "Burundi",
                "Chronological Bible Translation in Burundian Sign Language",
                "DOOR International",
            ],
            [
                "0a3ea85ee1e34a2d",
                "Nepali Sign Language",
                "nsp",
                "Nepal",
                "Chronological Bible Translation in Nepali Sign Language",
                "DOOR International",
            ],
            [
                "1fcac35962494d40",
                "Ugandan Sign Language",
                "ugn",
                "Uganda",
                "Chronological Bible Translation in Ugandan Sign Language",
                "DOOR International",
            ],
            [
                "a63aef8004db401b",
                "Ethiopian Sign Language",
                "eth",
                "Ethiopia",
                "Chronological Bible Translation in Ethiopian Sign Language",
                "DOOR International",
            ],
            [
                "56c922b7b5a44a47",
                "Kerala Sign Language",
                "mis",
                "India",
                "Chronological Bible Translation in Kerala Sign Language",
                "DOOR International",
            ],
            [
                "d35ef4f076de43f6",
                "Kerala Sign Language",
                "mis",
                "India",
                "Believe God How in Kerala Sign Language",
                "D.O.O.R. International",
            ],
            [
                "65c350c1cf9c42e4",
                "Nigerian Sign Language",
                "nsi",
                "Federal Republic Nigeria",
                "Chronological Bible Translation in Nigerian Sign Language",
                "DOOR International",
            ],
            [
                "6ad9cf57ab084a3b",
                "Estonian Sign Language",
                "eso",
                "Estonia",
                "The Bible in Estonian Sign Language",
                "Deaf Bible Society",
            ],
            [
                "9e9e20d036fa4e91",
                "Tanzanian Sign Language",
                "tza",
                "Tanzania",
                "Chronological Bible Translation in Tanzanian Sign Language",
                "DOOR International",
            ],
            [
                "6c1ffbf874d14ee1",
                "Andhra Pradesh Sign Language",
                "mis",
                "India",
                "Chronological Bible Translation in Andhra Pradesh Sign Language",
                "DOOR International",
            ],
            [
                "2d6ac5c8b4614955",
                "Bulgarian Sign Language",
                "bqn",
                "Bulgaria",
                "Chronological Bible Translation in Bulgarian Sign Language",
                "DOOR International",
            ],
            [
                "1bacaede20da4494",
                "American Sign Language",
                "ase",
                "United States of America",
                "Chronological Bible Translation in American Sign Language (119 Introductions and Passages expanded with More Information)",
                "Deaf Harbor ",
            ],
            [
                "d2027facd4cc4c2a",
                "American Sign Language",
                "ase",
                "United States of America",
                "Chronological Bible Translation in American Sign Language (119 Introductions and Passages)",
                "Deaf Harbor ",
            ],
            [
                "6543fec2ced7421d",
                "South Sudanese Sign Language",
                "mis",
                "Republic of South Sudan",
                "Chronological Bible Translation in South Sudanese Sign Language",
                "DOOR International",
            ],
            [
                "a28def50f139432a",
                "Kenyan Sign Language",
                "xki",
                "Kenya, Republic of",
                "Chronological Bible Translation in Kenyan Sign Language",
                "DOOR International",
            ],
            [
                "xki",
                "Kenya, Republic of",
                "Believe God How 52 in Kenyan Sign Language",
                "Deaf Opportunity Outreach International",
            ],
            [
                "c4b68657ce9b48ad",
                "Ghanaian Sign Language",
                "gse",
                "Ghana",
                "Chronological Bible Translation in Ghanaian Sign Language",
                "DOOR International",
            ],
            [
                "995240c9d7e8453e",
                "Indian (Delhi) Sign Language",
                "ins",
                "India",
                "Chronological Bible Translation in Indian (Delhi) Sign Language",
                "DOOR International",
            ],
            [
                "6d5944a5ceb944c0",
                "Russian Sign Language",
                "rsl",
                "Russian Federation",
                "Chronological Bible Translation in Russian Sign Language",
                "DOOR International",
            ],
            [
                "ec8517dba29d4d93",
                "Egyptian Sign Language",
                "esl",
                "Egypt",
                "Chronological Bible Translation in Egyptian Sign Language",
                "DOOR International",
            ],
            [
                "c0b48facec324e4b",
                "Mozambican Sign Language",
                "mzy",
                "Republic of Mozambique",
                "Chronological Bible Translation in Mozambican Sign Language",
                "DOOR International",
            ],
            [
                "b963267b41cc443c",
                "West Bengal Sign Language",
                "mis",
                "India",
                "Chronological Bible Translation in West Bengal Sign Language",
                "DOOR International",
            ],
            [
                "ae505f6ab3484407",
                "Tamil Nadu Sign Language",
                "mis",
                "India",
                "Chronological Bible Translation in Tamil Nadu Sign Language",
                "DOOR International",
            ],
        ]
    }
    return response_data


def refresh_manifest(output: Path = "manifest.json", auth=tuple | None):
    # Response data containing the list of sign language projects
    # TODO: check https://app.thedigitalbiblelibrary.org/entries/open_access_entries?type=video
    # TODO: figure out how to get this dynamically instead of hardcoded.
    response_data = get_response_data()

    # Create manifest
    manifest = create_manifest(response_data, auth)

    # Save manifest
    save_manifest(manifest, output)
    print(f"\nDone! Check {output.resolve()} for the results.")
    return output


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Generate a manifest of DBL sign language files.")
    parser.add_argument("--username", help="Username for DBL authentication")
    parser.add_argument("--password", help="Password for DBL authentication")
    parser.add_argument("--output", default="manifest.json", help="Output JSON file path")
    args = parser.parse_args()

    if args.username and args.password:
        auth = (args.username, args.password)
    else:
        print("Username and password are not provided, proceeding without them")
        auth = None
    refresh_manifest(output=args.output)
