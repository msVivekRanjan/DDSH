"""
download_haarcascades.py — Utility to download Haar Cascade Classifiers

This script downloads face and eye detection Haar Cascades from OpenCV GitHub
and places them in the haarcascades/ directory.

Run once before first training/detection attempt.
"""

import os
import urllib.request
import sys

# Haar cascade URLs from OpenCV repository
HAAR_CASCADES = {
    "haarcascade_frontalface_default.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
        "haarcascade_frontalface_default.xml"
    ),
    "haarcascade_eye.xml": (
        "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/"
        "haarcascade_eye.xml"
    ),
}

DESTINATION_DIR = "haarcascades"


def download_cascades() -> bool:
    """
    Download Haar cascade XML files from OpenCV repository.

    Returns:
        bool: True if all downloads successful, False otherwise.
    """

    print("\n" + "=" * 60)
    print("📥 Downloading Haar Cascade Classifiers")
    print("=" * 60)

    # Create haarcascades directory if not exists
    os.makedirs(DESTINATION_DIR, exist_ok=True)

    all_successful = True

    for filename, url in HAAR_CASCADES.items():
        file_path = os.path.join(DESTINATION_DIR, filename)

        # Skip if already exists
        if os.path.exists(file_path):
            print(f"✓ {filename} already exists")
            continue

        try:
            print(f"\n📥 Downloading {filename}...")
            print(f"   From: {url}")

            # Download file
            urllib.request.urlretrieve(url, file_path)

            # Verify file size (should be > 1 KB)
            file_size = os.path.getsize(file_path)
            if file_size > 1000:
                print(f"   ✓ Downloaded successfully ({file_size:,} bytes)")
            else:
                print(f"   ⚠️  Warning: File size unusually small ({file_size} bytes)")
                all_successful = False

        except Exception as e:
            print(f"   ❌ Failed to download: {str(e)}")
            all_successful = False

    print("\n" + "=" * 60)
    if all_successful:
        print("✅ All Haar cascades downloaded successfully!")
    else:
        print("⚠️  Some cascades failed. Check manual download below.")
        print("\n📌 Manual fallback:")
        print("   1. Visit: https://github.com/opencv/opencv/tree/master/data/haarcascades")
        print("   2. Download the .xml files")
        print("   3. Save to: haarcascades/")

    print("=" * 60 + "\n")

    return all_successful


def verify_cascades() -> bool:
    """
    Verify that Haar cascades are present and valid.

    Returns:
        bool: True if all cascades present, False otherwise.
    """

    print("\n✓ Verifying Haar cascades...")

    all_present = True
    for filename in HAAR_CASCADES.keys():
        file_path = os.path.join(DESTINATION_DIR, filename)
        if os.path.exists(file_path):
            print(f"  ✓ {filename}")
        else:
            print(f"  ❌ {filename} NOT FOUND")
            all_present = False

    return all_present


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DDSH — Haar Cascade Downloader Utility")
    print("=" * 60)

    # Download cascades
    download_cascades()

    # Verify
    if verify_cascades():
        print("\n✅ Setup complete! Ready to train and detect.")
        sys.exit(0)
    else:
        print("\n❌ Some cascades are missing. Manual download required.")
        sys.exit(1)
