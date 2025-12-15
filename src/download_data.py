"""
Download real data for Deep Learning in Asset Pricing

Downloads the dataset from the paper authors' Google Drive:
- Main page: https://mpelger.people.stanford.edu/data-and-code
- Data source: https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l

Usage:
    python -m src.download_data [--output_dir ./data]

    Or from Python:
        from src.download_data import download_all_data
        download_all_data("./data")
"""

import os
import sys
import shutil
import zipfile
import argparse
from pathlib import Path

try:
    import gdown
except ImportError:
    print("Error: 'gdown' package is required for downloading data.")
    print("Install it with: pip install gdown")
    sys.exit(1)


# Google Drive file ID for datasets.zip (contains all .npz files)
DATASETS_ZIP_ID = "1h9O7YwPLaRBbghtF50Cr-JmIq0aHHi4Y"

# Alternative: Download entire folder
GDRIVE_FOLDER_ID = "1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l"

# Expected file sizes for verification (approximate, in bytes)
EXPECTED_SIZES = {
    "Char_train.npz": 317 * 1024 * 1024,  # ~317 MB
    "Char_valid.npz": 72 * 1024 * 1024,    # ~72 MB
    "Char_test.npz": 768 * 1024 * 1024,    # ~768 MB
    "macro_train.npz": 351 * 1024,          # ~351 KB
    "macro_valid.npz": 96 * 1024,           # ~96 KB
    "macro_test.npz": 436 * 1024,           # ~436 KB
}


def check_data_exists(data_dir: str) -> dict:
    """
    Check which data files already exist.

    Args:
        data_dir: Path to data directory

    Returns:
        dict with 'missing' and 'existing' file lists
    """
    data_path = Path(data_dir)

    required_files = [
        data_path / "char" / "Char_train.npz",
        data_path / "char" / "Char_valid.npz",
        data_path / "char" / "Char_test.npz",
        data_path / "macro" / "macro_train.npz",
        data_path / "macro" / "macro_valid.npz",
        data_path / "macro" / "macro_test.npz",
    ]

    existing = [f for f in required_files if f.exists()]
    missing = [f for f in required_files if not f.exists()]

    return {
        "existing": existing,
        "missing": missing,
        "complete": len(missing) == 0
    }


def download_datasets_zip(output_dir: str, quiet: bool = False) -> bool:
    """
    Download datasets.zip from Google Drive and extract it.

    The zip contains:
        datasets/char/Char_train.npz, Char_valid.npz, Char_test.npz
        datasets/macro/macro_train.npz, macro_valid.npz, macro_test.npz

    Args:
        output_dir: Directory to save data to
        quiet: Suppress progress output

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    zip_path = output_path / "datasets.zip"
    url = f"https://drive.google.com/uc?id={DATASETS_ZIP_ID}"

    if not quiet:
        print("Downloading datasets.zip from Google Drive...")
        print(f"Source: {url}")
        print(f"Destination: {zip_path}")
        print()
        print("This may take a few minutes (~350 MB compressed, ~1.2 GB uncompressed)...")
        print()

    try:
        # Download the zip file
        gdown.download(url, str(zip_path), quiet=quiet)

        if not zip_path.exists():
            print("Error: Download failed - file not found")
            return False

        if not quiet:
            print()
            print("Extracting datasets.zip...")

        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(output_path)

        # Move files from datasets/ subdirectory to output_dir
        datasets_path = output_path / "datasets"
        if datasets_path.exists():
            # Create target directories
            (output_path / "char").mkdir(parents=True, exist_ok=True)
            (output_path / "macro").mkdir(parents=True, exist_ok=True)

            # Move char files
            char_src = datasets_path / "char"
            if char_src.exists():
                for f in char_src.glob("*.npz"):
                    dest = output_path / "char" / f.name
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(f), str(dest))
                    if not quiet:
                        print(f"  Extracted: char/{f.name}")

            # Move macro files
            macro_src = datasets_path / "macro"
            if macro_src.exists():
                for f in macro_src.glob("*.npz"):
                    dest = output_path / "macro" / f.name
                    if dest.exists():
                        dest.unlink()
                    shutil.move(str(f), str(dest))
                    if not quiet:
                        print(f"  Extracted: macro/{f.name}")

            # Clean up extracted datasets directory
            shutil.rmtree(datasets_path, ignore_errors=True)

            # Also remove __MACOSX if present
            macosx_path = output_path / "__MACOSX"
            if macosx_path.exists():
                shutil.rmtree(macosx_path, ignore_errors=True)

        # Optionally remove the zip file to save space
        if zip_path.exists():
            zip_path.unlink()
            if not quiet:
                print("  Removed: datasets.zip (no longer needed)")

        return True

    except zipfile.BadZipFile:
        print("Error: Downloaded file is not a valid zip archive")
        return False
    except Exception as e:
        print(f"Error: {e}")
        return False


def download_from_folder(output_dir: str, quiet: bool = False) -> bool:
    """
    Download all data files from the Google Drive folder.

    This downloads the entire folder, then extracts datasets.zip.

    Args:
        output_dir: Directory to save data to
        quiet: Suppress progress output

    Returns:
        True if successful, False otherwise
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}"

    if not quiet:
        print(f"Downloading data from Google Drive folder...")
        print(f"Source: {url}")
        print(f"Destination: {output_path.absolute()}")
        print()
        print("This may take several minutes...")
        print()

    try:
        gdown.download_folder(
            url=url,
            output=str(output_path),
            quiet=quiet,
            use_cookies=False
        )

        # Check if datasets.zip was downloaded and extract it
        zip_path = output_path / "datasets.zip"
        if zip_path.exists():
            if not quiet:
                print()
                print("Extracting datasets.zip...")

            with zipfile.ZipFile(zip_path, 'r') as zf:
                zf.extractall(output_path)

            # Move files from datasets/ to correct locations
            datasets_path = output_path / "datasets"
            if datasets_path.exists():
                (output_path / "char").mkdir(parents=True, exist_ok=True)
                (output_path / "macro").mkdir(parents=True, exist_ok=True)

                # Move char files
                char_src = datasets_path / "char"
                if char_src.exists():
                    for f in char_src.glob("*.npz"):
                        dest = output_path / "char" / f.name
                        if dest.exists():
                            dest.unlink()
                        shutil.move(str(f), str(dest))
                        if not quiet:
                            print(f"  Extracted: char/{f.name}")

                # Move macro files
                macro_src = datasets_path / "macro"
                if macro_src.exists():
                    for f in macro_src.glob("*.npz"):
                        dest = output_path / "macro" / f.name
                        if dest.exists():
                            dest.unlink()
                        shutil.move(str(f), str(dest))
                        if not quiet:
                            print(f"  Extracted: macro/{f.name}")

                # Clean up
                shutil.rmtree(datasets_path, ignore_errors=True)
                macosx_path = output_path / "__MACOSX"
                if macosx_path.exists():
                    shutil.rmtree(macosx_path, ignore_errors=True)

            # Remove zip to save space
            zip_path.unlink()
            if not quiet:
                print("  Removed: datasets.zip")

        return True
    except Exception as e:
        print(f"Error downloading folder: {e}")
        return False


def download_all_data(
    output_dir: str = "./data",
    force: bool = False,
    quiet: bool = False,
    method: str = "zip"
) -> bool:
    """
    Download all required data files for the Deep Learning Asset Pricing model.

    Args:
        output_dir: Directory to save data to (default: ./data)
        force: Re-download even if files exist
        quiet: Suppress progress output
        method: Download method - "zip" (recommended, faster) or "folder"

    Returns:
        True if all files downloaded successfully, False otherwise
    """
    output_path = Path(output_dir)

    # Check existing files
    if not force:
        status = check_data_exists(output_dir)
        if status["complete"]:
            if not quiet:
                print("All data files already exist!")
                print(f"Location: {output_path.absolute()}")
                print("\nUse --force to re-download.")
            return True
        elif status["existing"]:
            if not quiet:
                print(f"Found {len(status['existing'])} existing files.")
                print(f"Missing {len(status['missing'])} files:")
                for f in status['missing']:
                    print(f"  - {f}")
                print()

    # Create directories
    (output_path / "char").mkdir(parents=True, exist_ok=True)
    (output_path / "macro").mkdir(parents=True, exist_ok=True)

    if method == "zip":
        # Download just the datasets.zip (recommended - faster)
        success = download_datasets_zip(output_dir, quiet)
    else:
        # Download entire folder
        success = download_from_folder(output_dir, quiet)

    # Verify downloads
    if success:
        status = check_data_exists(output_dir)
        if status["complete"]:
            if not quiet:
                print()
                print("=" * 50)
                print("Download complete!")
                print("=" * 50)
                print()
                print("Data files saved to:")
                for f in status["existing"]:
                    size_mb = f.stat().st_size / (1024 * 1024)
                    print(f"  {f.relative_to(output_path)} ({size_mb:.1f} MB)")
                print()
                print("You can now train the model with:")
                print(f"  python -m src.train --data_dir {output_dir}")
            return True
        else:
            if not quiet:
                print()
                print("Warning: Some files may be missing after download.")
                print("Missing files:")
                for f in status["missing"]:
                    print(f"  - {f}")
                print()
                print("Try downloading manually from:")
                print(f"  https://drive.google.com/drive/folders/{GDRIVE_FOLDER_ID}")
            return False

    return False


def print_data_info():
    """Print information about the data files."""
    print("""
Deep Learning in Asset Pricing - Data Information
==================================================

The model requires the following data files (~1.2 GB total):

  data/
  ├── char/                      # Stock characteristics
  │   ├── Char_train.npz         (317 MB) - Training data
  │   ├── Char_valid.npz         (72 MB)  - Validation data
  │   └── Char_test.npz          (768 MB) - Test data
  └── macro/                     # Macroeconomic features
      ├── macro_train.npz        (351 KB)
      ├── macro_valid.npz        (96 KB)
      └── macro_test.npz         (436 KB)

Data Source:
  - Author's page: https://mpelger.people.stanford.edu/data-and-code
  - Google Drive: https://drive.google.com/drive/folders/1TrYzMUA_xLID5-gXOy_as8sH2ahLwz-l

Data Format (NPZ files):
  - Individual features: {data: [T, N, features+1], date: [T], variable: [features+1]}
    - data[:,:,0] contains stock returns
    - data[:,:,1:] contains firm characteristics
  - Macro features: {data: [T, macro_features], date: [T]}
    - Macroeconomic indicators over time
""")


def main():
    parser = argparse.ArgumentParser(
        description="Download data for Deep Learning in Asset Pricing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.download_data                    # Download to ./data
  python -m src.download_data --output_dir /path/to/data
  python -m src.download_data --force            # Re-download all files
  python -m src.download_data --info             # Show data information
        """
    )

    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        default="./data",
        help="Directory to save data files (default: ./data)"
    )

    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="Force re-download even if files exist"
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output"
    )

    parser.add_argument(
        "--info", "-i",
        action="store_true",
        help="Print information about data files and exit"
    )

    parser.add_argument(
        "--check",
        action="store_true",
        help="Check if data files exist and exit"
    )

    parser.add_argument(
        "--method", "-m",
        type=str,
        choices=["zip", "folder"],
        default="zip",
        help="Download method: 'zip' (faster, recommended) or 'folder' (default: zip)"
    )

    args = parser.parse_args()

    if args.info:
        print_data_info()
        return

    if args.check:
        status = check_data_exists(args.output_dir)
        if status["complete"]:
            print("All data files found!")
            for f in status["existing"]:
                size_mb = f.stat().st_size / (1024 * 1024)
                print(f"  {f} ({size_mb:.1f} MB)")
            sys.exit(0)
        else:
            print("Missing data files:")
            for f in status["missing"]:
                print(f"  {f}")
            sys.exit(1)

    # Download data
    success = download_all_data(
        output_dir=args.output_dir,
        force=args.force,
        quiet=args.quiet,
        method=args.method
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
