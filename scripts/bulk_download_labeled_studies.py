"""Optimized bulk downloader for RSNA labeled studies.

Caches Kaggle competition file listing once, filters for labeled studies,
builds a manifest, and downloads with bounded concurrency.

Features:
- Single Kaggle API call to cache all files (not per-study)
- Filters cached listing for 58 labeled StudyInstanceUIDs
- Builds .dcm manifest grouped by study/series
- Bounded concurrency (3-5 simultaneous downloads)
- DICOM validation, staged writes, retries, resumability

Usage:
  # Cache file listing and build manifest
  python scripts/bulk_download_labeled_studies.py --cache-only [--data-dir DIR]

  # Download 3 studies (test)
  python scripts/bulk_download_labeled_studies.py --dry-run  # Preview
  python scripts/bulk_download_labeled_studies.py --limit 3  # Download first 3

  # Download all 58
  python scripts/bulk_download_labeled_studies.py
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path
from queue import Queue

import pandas as pd

try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except ImportError:
    ThreadPoolExecutor = None


COMPETITION = "rsna-knee-abnormality-detection"
CACHE_FILE = Path("data/rsna-knee/kaggle_files_cache.json")
MAX_WORKERS = 4
MAX_RETRIES = 3
RETRY_BACKOFF = [1, 2, 4]


def load_simple_metadata(data_dir: Path):
    """Load train_series.csv without heavy dependencies."""
    root = Path(data_dir)
    train_series = pd.read_csv(
        root / "train_series.csv",
        dtype={"StudyInstanceUID": "string", "SeriesInstanceUID": "string"}
    )

    class SimpleMetadata:
        def __init__(self, series_df):
            self.train_series = series_df

        def series_for(self, study_uid: str, split: str = "train"):
            if split != "train":
                raise ValueError("Only train split supported")
            return self.train_series[self.train_series["StudyInstanceUID"] == str(study_uid)].copy()

    return SimpleMetadata(train_series)


def load_labeled_studies(data_dir: Path) -> set:
    """Get set of 58 labeled StudyInstanceUIDs."""
    md = load_simple_metadata(data_dir)
    labeled = md.train_series  # All rows from train_series.csv

    # Read train.csv to find which studies have all labels
    train_file = Path(data_dir) / "train.csv"
    train_df = pd.read_csv(train_file, dtype={"StudyInstanceUID": "string"})

    target_cols = [
        "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
        "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
    ]

    # Studies with all labels (no NaN in any target column)
    complete_labeled = train_df.dropna(subset=target_cols)
    return set(complete_labeled["StudyInstanceUID"].astype(str).tolist())


def is_valid_dicom(path: Path) -> bool:
    """Validate DICOM by checking magic bytes at offset 128."""
    if not path.is_file() or path.stat().st_size < 132:
        return False
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except:
        return False


def fetch_and_cache_file_listing(data_dir: Path, cache_file: Path = CACHE_FILE) -> dict:
    """Fetch competition file listing from Kaggle and cache locally.

    Returns: {study_uid: {series_uid: [file_paths]}}
    """

    print("Fetching Kaggle file listing (this takes ~10-15 minutes)...")
    print("(Querying ~300 pages with rate limiting)")

    all_files = defaultdict(lambda: defaultdict(list))

    cmd = ["kaggle", "competitions", "files", COMPETITION, "-v", "--page-size", "100"]

    page_token = None
    pages = 0

    while pages < 500:  # Safety limit
        try:
            if page_token:
                cmd_with_token = cmd + ["--page-token", page_token]
            else:
                cmd_with_token = cmd

            result = subprocess.run(
                cmd_with_token,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode != 0:
                error_msg = result.stderr or result.stdout
                if "429" in error_msg or "Too Many Requests" in error_msg:
                    wait_time = min(60, 5 + pages // 50)  # Exponential backoff
                    print(f"  Rate limited on page {pages}, waiting {wait_time}s...", flush=True)
                    time.sleep(wait_time)
                    continue
                elif not error_msg:
                    # Empty error - might be transient, retry
                    print(f"  Empty response on page {pages}, retrying...", flush=True)
                    time.sleep(5)
                    continue
                raise Exception(f"Kaggle CLI error: {error_msg[:200]}")

            lines = result.stdout.strip().split('\n')
            pages += 1

            # Aggressive rate limiting: delay after every request
            time.sleep(2)  # Always wait 2s between requests

            if pages % 20 == 0:
                print(f"  Page {pages}...", flush=True)

            # Extract next page token
            next_token = None
            for line in lines:
                if line.startswith("Next Page Token = "):
                    next_token = line.replace("Next Page Token = ", "").strip()
                    break

            # Collect train_series files
            for line in lines:
                if "train_series" in line and ".dcm" in line:
                    parts = line.split(',')
                    if len(parts) >= 1:
                        file_path = parts[0].strip()
                        path_parts = file_path.split('/')
                        if len(path_parts) >= 3:
                            study_uid = path_parts[1]
                            series_uid = path_parts[2]
                            all_files[study_uid][series_uid].append(file_path)

            if not next_token:
                break

            page_token = next_token

        except subprocess.TimeoutExpired:
            raise Exception("Kaggle CLI timeout")
        except Exception as e:
            raise Exception(f"Fetch failed: {e}")

    print(f"\nCached {len(all_files)} studies, {sum(len(s) for s in all_files.values())} series")

    # Save cache
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_data = {
        study: {series: files for series, files in series_files.items()}
        for study, series_files in all_files.items()
    }
    with open(cache_file, 'w') as f:
        json.dump(cache_data, f)

    print(f"Cache saved to: {cache_file}")
    print(f"Cache size: {cache_file.stat().st_size / (1024*1024):.1f} MB")

    return all_files


def filter_for_labeled_studies(all_files: dict, labeled_studies: set) -> dict:
    """Filter cached file listing for labeled studies only."""
    labeled_files = {
        study: series_files
        for study, series_files in all_files.items()
        if study in labeled_studies
    }

    print(f"\nFiltered: {len(labeled_files)} labeled studies found")
    print(f"  Total series: {sum(len(s) for s in labeled_files.values())}")
    print(f"  Total files: {sum(len(f) for s in labeled_files.values() for f in s.values())}")

    return labeled_files


def build_manifest(labeled_files: dict, data_dir: Path) -> dict:
    """Build manifest with study/series/file organization and metadata verification."""
    md = load_simple_metadata(data_dir)

    manifest = {}
    errors = []

    for study_uid in sorted(labeled_files.keys()):
        series_files = labeled_files[study_uid]

        # Verify against metadata
        metadata_series = set(
            md.series_for(study_uid, "train").SeriesInstanceUID.astype(str).tolist()
        )
        discovered_series = set(series_files.keys())

        missing = metadata_series - discovered_series
        extra = discovered_series - metadata_series

        if missing or extra:
            errors.append(f"{study_uid[-12:]}: missing {len(missing)}, extra {len(extra)}")

        manifest[study_uid] = {
            'series': {
                series_uid: {
                    'files': files,
                    'count': len(files),
                }
                for series_uid, files in series_files.items()
            },
            'metadata_valid': len(missing) == 0 and len(extra) == 0,
        }

    if errors:
        print(f"\nWarnings:")
        for err in errors[:5]:
            print(f"  {err}")

    return manifest


def download_file_with_retry(file_path: str, destination: Path) -> tuple[bool, str]:
    """Download one DICOM file with retry logic."""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with tempfile.TemporaryDirectory(prefix="dcm_") as staging_dir:
                staging = Path(staging_dir)

                cmd = [
                    "kaggle", "competitions", "download",
                    COMPETITION,
                    "-f", file_path,
                    "-p", str(staging),
                    "-q"
                ]

                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if result.returncode != 0:
                    error_msg = (result.stderr or result.stdout).strip()[:100]

                    is_transient = any(x in error_msg.lower() for x in [
                        "timeout", "connection", "temporarily", "429", "503", "502"
                    ])

                    if is_transient and attempt < MAX_RETRIES:
                        time.sleep(RETRY_BACKOFF[attempt - 1])
                        continue
                    else:
                        return False, error_msg

                dcm_files = list(staging.rglob("*.dcm"))
                if not dcm_files:
                    return False, "No .dcm file in download"

                downloaded = dcm_files[0]

                if not is_valid_dicom(downloaded):
                    return False, "Downloaded file is not valid DICOM"

                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(downloaded), str(destination))
                return True, None

        except subprocess.TimeoutExpired:
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF[attempt - 1])
                continue
            return False, "Timeout after retries"
        except Exception as e:
            return False, str(e)[:100]

    return False, f"Failed after {MAX_RETRIES} attempts"


def download_studies_concurrent(
    studies_to_download: list,
    labeled_files: dict,
    output_root: Path,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS
) -> dict:
    """Download multiple studies with concurrent file downloads."""

    results = {}
    start_time = time.time()

    for study_uid in studies_to_download:
        study_result = download_single_study(
            study_uid,
            labeled_files[study_uid],
            output_root,
            dry_run=dry_run,
            max_workers=max_workers
        )
        results[study_uid] = study_result

    elapsed = time.time() - start_time

    return {
        'studies': results,
        'elapsed_seconds': elapsed,
        'total_downloaded': sum(r['downloaded'] for r in results.values()),
        'total_skipped': sum(r['skipped'] for r in results.values()),
        'total_failed': sum(r['failed'] for r in results.values()),
        'total_size_mb': sum(r['size_mb'] for r in results.values()),
    }


def download_single_study(
    study_uid: str,
    series_files: dict,
    output_root: Path,
    dry_run: bool = False,
    max_workers: int = MAX_WORKERS
) -> dict:
    """Download all files for one study with concurrent downloads."""

    stats = {
        'study_uid': study_uid,
        'series': {},
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'size_mb': 0.0,
        'errors': [],
    }

    print(f"\nStudy {study_uid[-12:]}:")

    if dry_run:
        total_files = sum(len(files) for files in series_files.values())
        print(f"  [DRY-RUN] Would download {total_files} files across {len(series_files)} series")
        for series_uid, files in series_files.items():
            stats['series'][series_uid] = {
                'downloaded': 0,
                'skipped': 0,
                'failed': 0,
                'size_mb': 0.0,
                'files': len(files),
            }
        return stats

    # Check what's already downloaded
    study_dir = output_root / study_uid
    existing_valid = {}
    for series_uid, files in series_files.items():
        series_dir = study_dir / series_uid
        existing_valid[series_uid] = set()
        if series_dir.exists():
            for file_path in files:
                filename = file_path.split('/')[-1]
                local_path = series_dir / filename
                if local_path.exists() and is_valid_dicom(local_path):
                    existing_valid[series_uid].add(filename)

    # Build download queue
    download_queue = []
    for series_uid, files in series_files.items():
        series_dest = study_dir / series_uid
        for file_path in files:
            filename = file_path.split('/')[-1]
            if filename not in existing_valid[series_uid]:
                download_queue.append((file_path, series_dest / filename))

    skipped_per_series = {
        series_uid: len(existing_valid[series_uid])
        for series_uid in series_files.keys()
    }

    # Download with bounded concurrency
    if download_queue and ThreadPoolExecutor:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(download_file_with_retry, file_path, dest): (file_path, dest)
                for file_path, dest in download_queue
            }

            for future in as_completed(futures):
                file_path, dest = futures[future]
                series_uid = dest.parent.name

                try:
                    success, error = future.result()
                    if success:
                        stats['downloaded'] += 1
                        stats['size_mb'] += dest.stat().st_size / (1024 * 1024)
                    else:
                        stats['failed'] += 1
                        stats['errors'].append(f"{series_uid[-12:]}: {error}")
                except Exception as e:
                    stats['failed'] += 1
                    stats['errors'].append(f"{series_uid[-12:]}: {str(e)[:100]}")

    # Summarize per-series
    for series_uid in series_files.keys():
        series_dir = study_dir / series_uid
        final_count = sum(
            1 for f in series_dir.glob("*.dcm") if is_valid_dicom(f)
        ) if series_dir.exists() else 0

        stats['series'][series_uid] = {
            'total_files': len(series_files[series_uid]),
            'downloaded': sum(1 for f in download_queue if f[1].parent.name == series_uid),
            'skipped': skipped_per_series[series_uid],
            'final_valid': final_count,
        }

        stats['skipped'] += skipped_per_series[series_uid]

    # Print summary
    total_files = sum(len(files) for files in series_files.values())
    print(f"  Files: {stats['downloaded']} downloaded + {stats['skipped']} existing "
          f"({stats['size_mb']:.1f} MB) [{stats['failed']} failed]")

    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/rsna-knee"),
                        help="Dataset directory")
    parser.add_argument("--cache-only", action="store_true",
                        help="Fetch and cache file listing only, don't download")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview downloads without fetching files")
    parser.add_argument("--limit", type=int, default=None,
                        help="Download only N studies (for testing)")
    parser.add_argument("--max-workers", type=int, default=MAX_WORKERS,
                        help="Concurrent downloads")

    args = parser.parse_args(argv)

    try:
        # Load labeled studies
        labeled_studies = load_labeled_studies(args.data_dir)
        print(f"Found {len(labeled_studies)} labeled studies in metadata\n")

        # Check for cached listing
        if CACHE_FILE.exists() and not args.cache_only:
            print(f"Using cached file listing: {CACHE_FILE}")
            with open(CACHE_FILE) as f:
                all_files = json.load(f)
            print(f"  Studies: {len(all_files)}, Size: {CACHE_FILE.stat().st_size / (1024*1024):.1f} MB\n")
        else:
            # Fetch and cache
            all_files = fetch_and_cache_file_listing(args.data_dir)

        if args.cache_only:
            return 0

        # Filter for labeled studies
        labeled_files = filter_for_labeled_studies(all_files, labeled_studies)

        # Build manifest
        manifest = build_manifest(labeled_files, args.data_dir)

        # Determine studies to download
        studies = sorted(labeled_files.keys())
        if args.limit:
            studies = studies[:args.limit]

        print(f"\nDownloading {len(studies)} studies...")

        # Download
        output_root = Path(args.data_dir) / "raw" / "train_series"
        result = download_studies_concurrent(
            studies,
            labeled_files,
            output_root,
            dry_run=args.dry_run,
            max_workers=args.max_workers
        )

        # Report
        print("\n" + "=" * 90)
        print("BULK DOWNLOAD SUMMARY")
        print("=" * 90)

        for study_uid, study_result in result['studies'].items():
            print(f"\n{study_uid[-12:]}:")
            for series_uid, s_stats in study_result['series'].items():
                print(f"  {series_uid[-12:]}: {s_stats.get('final_valid', 0)} valid DICOMs")

        print(f"\nTotals:")
        print(f"  Downloaded: {result['total_downloaded']}")
        print(f"  Skipped: {result['total_skipped']}")
        print(f"  Failed: {result['total_failed']}")
        print(f"  Size: {result['total_size_mb']:.1f} MB")
        print(f"  Time: {result['elapsed_seconds']:.1f}s")
        print("=" * 90)

        return 0 if result['total_failed'] == 0 else 1

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
