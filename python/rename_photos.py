#!/usr/bin/env python3
"""
Photo & Video Renamer Tool
--------------------------
Renames photo and video files based on their EXIF capture timestamp and closest city.

Filename format:
  - With GPS / Geotagging:  yyyymmdd_hhmmss_city.ext  (e.g., 20260729_151057_Copenhagen.jpg)
  - Without Geotagging:     yyyymmdd_hhmmss.ext       (e.g., 20260729_151057.jpg)

Features:
  - Extracts true capture date from EXIF (DateTimeOriginal / CreateDate / SubSecDateTime)
  - Reverse geocodes GPS coordinates to closest city in English (via OpenStreetMap Nominatim with local caching)
  - Automatic duplicate/burst handling (appends _1, _2, etc. to prevent collisions)
  - Supports JPEG, PNG, TIFF, HEIC, MP4, MOV, DNG, CR2, NEF, ARW, and other formats
  - Safe dry-run mode (--dry-run / -n)
  - ASCII transliteration option (--ascii) for Danish, German, and European characters
  - Optional fallback city (--city / --default-city) for photos without GPS data
  - Built-in geo test utility (--test-geo LAT LON)
"""

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
import unicodedata
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

SUPPORTED_EXTENSIONS = {
    # Images
    '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.heic', '.heif', '.webp',
    '.raw', '.cr2', '.cr3', '.nef', '.arw', '.dng', '.orf', '.rw2', '.pef',
    # Videos
    '.mp4', '.mov', '.avi', '.m4v', '.3gp', '.mkv'
}

DEFAULT_CACHE_FILE = Path.home() / ".cache" / "photo_renamer_geocache.json"


class GeoCityResolver:
    """Resolves GPS coordinates to city names in English with spatial caching and rate limiting."""

    def __init__(
        self,
        cache_file: Optional[Path] = None,
        ascii_only: bool = False,
        lang: str = "en",
        verbose: bool = False
    ):
        self.cache_file = cache_file or DEFAULT_CACHE_FILE
        self.ascii_only = ascii_only
        self.lang = lang or "en"
        self.verbose = verbose
        self.cache: Dict[str, Dict] = self._load_cache()
        self.last_request_time = 0.0

    def _load_cache(self) -> Dict[str, Dict]:
        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Warning: Failed to load geo cache from {self.cache_file}: {e}")
        return {}

    def _save_cache(self):
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to save geo cache: {e}")

    @staticmethod
    def _haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate the great circle distance in km between two points."""
        r = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return r * c

    def _find_nearby_cached(self, lat: float, lon: float, max_dist_km: float = 1.5) -> Optional[str]:
        """Check if a coordinate within max_dist_km is already in the cache for the same language."""
        for key, val in self.cache.items():
            try:
                coord_part = key.split(":")[0]
                lang_part = key.split(":")[1] if ":" in key else "en"
                if lang_part != self.lang:
                    continue
                c_lat, c_lon = map(float, coord_part.split(","))
                dist = self._haversine_distance_km(lat, lon, c_lat, c_lon)
                if dist <= max_dist_km and val.get("city"):
                    return val["city"]
            except ValueError:
                continue
        return None

    def clean_administrative_names(self, name: str) -> str:
        """Strip administrative words like 'Municipality', 'Kommune', 'City of'."""
        if not name:
            return ""
        name = re.sub(r'\s+(Municipality|Kommune|County|District|Gemeinde|Stadt)$', '', name, flags=re.IGNORECASE)
        name = re.sub(r'^(City of|Town of|Village of|Municipality of)\s+', '', name, flags=re.IGNORECASE)
        return name.strip()

    def sanitize_city(self, name: str) -> str:
        """Sanitize city name for filesystem compatibility."""
        if not name:
            return ""

        name = self.clean_administrative_names(name)

        if self.ascii_only:
            # Custom European character replacements
            charmap = {
                'ø': 'o', 'Ø': 'O',
                'æ': 'ae', 'Æ': 'Ae',
                'å': 'aa', 'Å': 'Aa',
                'ä': 'ae', 'Ä': 'Ae',
                'ö': 'oe', 'Ö': 'Oe',
                'ü': 'ue', 'Ü': 'Ue',
                'ß': 'ss',
                'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
                'á': 'a', 'à': 'a', 'â': 'a',
                'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
                'ó': 'o', 'ò': 'o', 'ô': 'o',
                'ú': 'u', 'ù': 'u', 'û': 'u',
                'ñ': 'n', 'ç': 'c',
            }
            for k, v in charmap.items():
                name = name.replace(k, v)
            name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('ascii')
        else:
            name = unicodedata.normalize('NFC', name)

        # Replace spaces, slashes, hyphens with underscores
        name = re.sub(r'[\s/\\|]+', '_', name.strip())
        # Remove filesystem illegal characters
        name = re.sub(r'[:*?"<>#%&{}\\<>*?/$!\'":@;,]', '', name)
        # Collapse repeated underscores
        name = re.sub(r'_+', '_', name)
        return name.strip('_')

    def get_city(self, lat: float, lon: float) -> Optional[str]:
        """Resolve (lat, lon) to a city name in English."""
        if lat is None or lon is None:
            return None

        # Cache key rounded to ~11m precision including language code
        cache_key = f"{lat:.4f},{lon:.4f}:{self.lang}"
        if cache_key in self.cache:
            raw_city = self.cache[cache_key].get("city")
            return self.sanitize_city(raw_city) if raw_city else None

        # Check nearby cache (within 1.5 km)
        nearby_city = self._find_nearby_cached(lat, lon, max_dist_km=1.5)
        if nearby_city:
            self.cache[cache_key] = {"city": nearby_city, "source": "nearby_cache"}
            self._save_cache()
            return self.sanitize_city(nearby_city)

        # Online reverse geocoding via OpenStreetMap Nominatim with English preference
        elapsed = time.time() - self.last_request_time
        if elapsed < 1.1:
            time.sleep(1.1 - elapsed)

        headers = {
            "User-Agent": "PhotoRenamerScript/1.0 (photo_rename_tool)",
            "Accept-Language": self.lang
        }

        url = (
            f"https://nominatim.openstreetmap.org/reverse"
            f"?lat={lat}&lon={lon}&format=json&zoom=14&addressdetails=1"
        )
        req = urllib.request.Request(url, headers=headers)

        try:
            self.last_request_time = time.time()
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                addr = data.get("address", {})
                city = (
                    addr.get("city")
                    or addr.get("town")
                    or addr.get("village")
                    or addr.get("hamlet")
                    or addr.get("suburb")
                    or addr.get("municipality")
                    or addr.get("county")
                )
                if city:
                    city_clean = self.clean_administrative_names(city)
                    self.cache[cache_key] = {"city": city_clean, "source": "nominatim"}
                    self._save_cache()
                    return self.sanitize_city(city_clean)
                else:
                    self.cache[cache_key] = {"city": "", "source": "nominatim_empty"}
                    self._save_cache()
        except Exception as e:
            if self.verbose:
                print(f"Geocoding request failed for ({lat}, {lon}): {e}")

        return None


class ExifExtractor:
    """Extracts date/time and GPS coordinates from images and videos."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.has_exiftool = shutil.which("exiftool") is not None

    def batch_extract_exiftool(self, file_paths: List[Path]) -> Dict[Path, Dict]:
        """Batch extract EXIF metadata using exiftool."""
        results = {}
        if not file_paths or not self.has_exiftool:
            return results

        chunk_size = 200
        for i in range(0, len(file_paths), chunk_size):
            chunk = file_paths[i:i + chunk_size]
            cmd = [
                "exiftool",
                "-j",
                "-n",
                "-q",
                "-DateTimeOriginal",
                "-CreateDate",
                "-SubSecDateTimeOriginal",
                "-SubSecCreateDate",
                "-MediaCreateDate",
                "-TrackCreateDate",
                "-ModifyDate",
                "-GPSLatitude",
                "-GPSLongitude",
                "-GPSPosition",
            ] + [str(p) for p in chunk]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
                data = json.loads(proc.stdout)
                for item in data:
                    src = Path(item.get("SourceFile", "")).resolve()
                    dt = self._parse_exiftool_date(item)
                    lat = item.get("GPSLatitude")
                    lon = item.get("GPSLongitude")
                    if lat is not None and lon is not None:
                        try:
                            lat = float(lat)
                            lon = float(lon)
                        except (ValueError, TypeError):
                            lat, lon = None, None

                    results[src] = {
                        "datetime": dt,
                        "lat": lat,
                        "lon": lon,
                    }
            except Exception as e:
                if self.verbose:
                    print(f"Exiftool batch execution error: {e}")

        return results

    def _parse_exiftool_date(self, item: Dict) -> Optional[datetime.datetime]:
        date_keys = [
            "DateTimeOriginal",
            "CreateDate",
            "SubSecDateTimeOriginal",
            "SubSecCreateDate",
            "MediaCreateDate",
            "TrackCreateDate",
            "ModifyDate"
        ]
        for k in date_keys:
            val = item.get(k)
            if val:
                dt = self.parse_date_string(str(val))
                if dt:
                    return dt
        return None

    def extract_single(self, file_path: Path) -> Tuple[Optional[datetime.datetime], Optional[float], Optional[float]]:
        """Extract metadata for a single file using Pillow / filename fallback."""
        dt = None
        lat = None
        lon = None

        # 1. Try Pillow
        try:
            from PIL import Image, ExifTags
            with Image.open(file_path) as img:
                exif = img.getexif()
                if exif:
                    for tag_id in (36867, 36868, 306):
                        if tag_id in exif:
                            dt = self.parse_date_string(str(exif[tag_id]))
                            if dt:
                                break

                    if not dt and hasattr(ExifTags, 'IFD'):
                        try:
                            exif_ifd = exif.get_ifd(ExifTags.IFD.Exif)
                            for tag_id in (36867, 36868):
                                if tag_id in exif_ifd:
                                    dt = self.parse_date_string(str(exif_ifd[tag_id]))
                                    if dt:
                                        break
                        except Exception:
                            pass

                    try:
                        gps_ifd = None
                        if hasattr(ExifTags, 'IFD'):
                            gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
                        if gps_ifd:
                            lat, lon = self._parse_pillow_gps(gps_ifd)
                    except Exception:
                        pass
        except Exception:
            pass

        # 2. Fallback to filename timestamp
        if not dt:
            dt = self.extract_date_from_filename(file_path.name)

        # 3. Fallback to file modification time
        if not dt:
            try:
                mtime = os.path.getmtime(file_path)
                dt = datetime.datetime.fromtimestamp(mtime)
            except Exception:
                pass

        return dt, lat, lon

    @staticmethod
    def _parse_pillow_gps(gps_ifd: Dict) -> Tuple[Optional[float], Optional[float]]:
        def to_deg(coord):
            if not coord or len(coord) < 3:
                return None
            d = float(coord[0])
            m = float(coord[1])
            s = float(coord[2])
            return d + (m / 60.0) + (s / 3600.0)

        lat_ref = gps_ifd.get(1, 'N')
        lat_raw = gps_ifd.get(2)
        lon_ref = gps_ifd.get(3, 'E')
        lon_raw = gps_ifd.get(4)

        if lat_raw and lon_raw:
            lat = to_deg(lat_raw)
            lon = to_deg(lon_raw)
            if lat is not None and str(lat_ref).upper() == 'S':
                lat = -lat
            if lon is not None and str(lon_ref).upper() == 'W':
                lon = -lon
            return lat, lon
        return None, None

    @staticmethod
    def parse_date_string(date_str: str) -> Optional[datetime.datetime]:
        if not date_str:
            return None
        cleaned = date_str.strip()
        m = re.match(r'^(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2})[:\-](\d{2})[:\-](\d{2})', cleaned)
        if m:
            try:
                return datetime.datetime(
                    year=int(m.group(1)),
                    month=int(m.group(2)),
                    day=int(m.group(3)),
                    hour=int(m.group(4)),
                    minute=int(m.group(5)),
                    second=int(m.group(6))
                )
            except ValueError:
                pass
        return None

    @staticmethod
    def extract_date_from_filename(filename: str) -> Optional[datetime.datetime]:
        patterns = [
            r'(\d{4})(\d{2})(\d{2})_(\d{2})(\d{2})(\d{2})',
            r'(\d{4})-(\d{2})-(\d{2})_(\d{2})-(\d{2})-(\d{2})',
            r'(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})',
        ]
        for pat in patterns:
            m = re.search(pat, filename)
            if m:
                try:
                    return datetime.datetime(
                        year=int(m.group(1)),
                        month=int(m.group(2)),
                        day=int(m.group(3)),
                        hour=int(m.group(4)),
                        minute=int(m.group(5)),
                        second=int(m.group(6))
                    )
                except ValueError:
                    continue
        return None


def build_rename_plan(
    files: List[Path],
    geo_resolver: Optional[GeoCityResolver],
    use_city: bool = True,
    default_city: Optional[str] = None,
    verbose: bool = False
) -> Tuple[List[Tuple[Path, Path, str]], int]:
    """
    Builds a rename mapping (src, target, status) with collision avoidance.
    Returns (plan, files_with_gps_count).
    """
    extractor = ExifExtractor(verbose=verbose)
    exif_batch = {}

    if extractor.has_exiftool:
        if verbose:
            print(f"Extracting metadata using exiftool for {len(files)} files...")
        exif_batch = extractor.batch_extract_exiftool(files)

    plan = []
    used_targets: Dict[Path, int] = {}
    gps_count = 0

    for src in files:
        src_resolved = src.resolve()
        info = exif_batch.get(src_resolved)
        if info:
            dt = info.get("datetime")
            lat = info.get("lat")
            lon = info.get("lon")
        else:
            dt, lat, lon = extractor.extract_single(src)

        if not dt:
            plan.append((src, src, "Skipped: Unable to determine date/time"))
            continue

        timestamp_str = dt.strftime("%Y%m%d_%H%M%S")

        city_name = ""
        if use_city and geo_resolver and lat is not None and lon is not None:
            gps_count += 1
            resolved = geo_resolver.get_city(lat, lon)
            if resolved:
                city_name = resolved
                if verbose:
                    print(f"File {src.name} (GPS {lat:.4f}, {lon:.4f}) -> {city_name}")
        elif default_city:
            city_name = geo_resolver.sanitize_city(default_city) if geo_resolver else default_city

        if city_name:
            base_name = f"{timestamp_str}_{city_name}"
        else:
            base_name = f"{timestamp_str}"

        ext = src.suffix.lower()
        parent = src.parent

        target = parent / f"{base_name}{ext}"
        counter = 1

        while True:
            is_same_file = False
            try:
                if target.exists() and src.samefile(target):
                    is_same_file = True
            except Exception:
                pass

            if target in used_targets or (target.exists() and not is_same_file):
                target = parent / f"{base_name}_{counter}{ext}"
                counter += 1
            else:
                break

        used_targets[target] = 1

        if src.name == target.name:
            plan.append((src, target, "Unchanged (already matches format)"))
        else:
            plan.append((src, target, "Rename"))

    return plan, gps_count


def main():
    parser = argparse.ArgumentParser(
        description="Rename photos based on EXIF time and closest city in English (format: yyyymmdd_hhmmss_city.ext)."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        default=["."],
        help="Directories or files to process (default: current directory)."
    )
    parser.add_argument(
        "-n", "--dry-run",
        action="store_true",
        help="Preview proposed renames without making any changes."
    )
    parser.add_argument(
        "-y", "--yes",
        action="store_true",
        help="Execute renaming without asking for confirmation."
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="Process directories recursively."
    )
    parser.add_argument(
        "--no-city",
        action="store_true",
        help="Do not look up city names, even if GPS data is present."
    )
    parser.add_argument(
        "--city", "--default-city",
        type=str,
        default=None,
        dest="default_city",
        help="Fallback city name to use when photos do not contain GPS metadata (e.g. --city Copenhagen)."
    )
    parser.add_argument(
        "--ascii",
        action="store_true",
        help="Convert city names to plain ASCII (e.g. København -> Kobenhavn)."
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Preferred language code for city names (default: 'en')."
    )
    parser.add_argument(
        "-c", "--copy",
        action="store_true",
        help="Copy files to new names instead of moving/renaming them."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output."
    )
    parser.add_argument(
        "--test-geo",
        nargs=2,
        metavar=("LAT", "LON"),
        type=float,
        help="Test reverse geocoding for specific latitude and longitude and exit."
    )

    args = parser.parse_args()

    # Handle test-geo mode
    if args.test_geo:
        lat, lon = args.test_geo
        print(f"Testing English reverse geocoding for ({lat}, {lon})...")
        resolver = GeoCityResolver(ascii_only=args.ascii, lang=args.lang, verbose=True)
        city = resolver.get_city(lat, lon)
        print(f"Result: City = '{city}' (Language: {args.lang})")
        return

    # Collect all target files
    target_files: List[Path] = []
    for p_str in args.paths:
        p = Path(p_str)
        if p.is_file():
            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                target_files.append(p)
        elif p.is_dir():
            pattern = "**/*" if args.recursive else "*"
            for f in p.glob(pattern):
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS and not f.name.startswith("."):
                    target_files.append(f)

    target_files.sort()

    if not target_files:
        print("No supported photo or video files found to process.")
        return

    print(f"Found {len(target_files)} media files to evaluate.")

    geo_resolver = GeoCityResolver(
        ascii_only=args.ascii,
        lang=args.lang,
        verbose=args.verbose
    )

    plan, gps_count = build_rename_plan(
        target_files,
        geo_resolver=geo_resolver,
        use_city=not args.no_city,
        default_city=args.default_city,
        verbose=args.verbose
    )

    renames_to_do = [item for item in plan if item[2] == "Rename"]
    unchanged = [item for item in plan if "Unchanged" in item[2]]
    skipped = [item for item in plan if "Skipped" in item[2]]

    print("\n" + "=" * 80)
    print(f"{'CURRENT NAME':<38} -> {'NEW NAME':<38}")
    print("=" * 80)

    display_limit = 50 if (not args.verbose and len(plan) > 60) else len(plan)
    for src, target, status in plan[:display_limit]:
        if status == "Rename":
            print(f"{src.name:<38} -> {target.name:<38}")
        elif args.verbose:
            print(f"{src.name:<38} == {target.name:<38} ({status})")

    if len(plan) > display_limit:
        print(f"... and {len(plan) - display_limit} more files (run with --verbose to view all).")

    print("=" * 80)
    print(f"Summary: {len(renames_to_do)} to rename, {len(unchanged)} already correct, {len(skipped)} skipped.")
    print(f"Geotagging: {gps_count}/{len(target_files)} files contained GPS coordinates.")
    if gps_count == 0 and not args.default_city:
        print("Note: None of the evaluated files contained embedded GPS coordinates.")
        print("      To set a city for these photos, use the --city option (e.g., --city Copenhagen).")

    if args.dry_run:
        print("\n[DRY RUN] No files were modified. Run without --dry-run to apply changes.")
        return

    if not renames_to_do:
        print("\nAll files are already properly named.")
        return

    # If running interactively and -y not provided, confirm
    if not args.yes and sys.stdin.isatty():
        try:
            confirm = input(f"\nProceed with {'copying' if args.copy else 'renaming'} {len(renames_to_do)} files? [y/N]: ").strip().lower()
            if confirm not in ("y", "yes"):
                print("Aborted by user.")
                return
        except KeyboardInterrupt:
            print("\nAborted.")
            return

    action_word = "Copying" if args.copy else "Renaming"
    print(f"\n{action_word} {len(renames_to_do)} files...")

    success_count = 0
    error_count = 0

    for src, target, status in renames_to_do:
        try:
            if args.copy:
                shutil.copy2(src, target)
            else:
                src.rename(target)
            success_count += 1
        except Exception as e:
            print(f"Error {action_word.lower()} {src.name} -> {target.name}: {e}")
            error_count += 1

    print(f"Done! Successfully processed {success_count} files ({error_count} errors).")


if __name__ == "__main__":
    main()
