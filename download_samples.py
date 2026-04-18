"""
Download EXIF sample images for demo.
Sources: ianare/exif-samples (GitHub) — public domain test files with GPS + device metadata.
"""

import os
import requests

SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "sample_data")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# Raw URLs from ianare/exif-samples — public repo, all public domain test files
# These are standard EXIF test images used by exif libraries worldwide
SAMPLES = [
    # GPS-bearing images — confirmed GPS coordinates in EXIF
    ("jpg/gps/DSCN0010.jpg",  "GPS_01_DSCN0010_Nikon.jpg"),
    ("jpg/gps/DSCN0012.jpg",  "GPS_02_DSCN0012_Nikon.jpg"),
    ("jpg/gps/DSCN0021.jpg",  "GPS_03_DSCN0021_Nikon.jpg"),
    ("jpg/gps/DSCN0025.jpg",  "GPS_04_DSCN0025_Nikon.jpg"),
    ("jpg/gps/DSCN0027.jpg",  "GPS_05_DSCN0027_Nikon.jpg"),
    ("jpg/gps/DSCN0029.jpg",  "GPS_06_DSCN0029_Nikon.jpg"),
    ("jpg/gps/DSCN0038.jpg",  "GPS_07_DSCN0038_Nikon.jpg"),
    ("jpg/gps/DSCN0042.jpg",  "GPS_08_DSCN0042_Nikon.jpg"),
    # Non-GPS images — good for "metadata stripped" contrast in demo
    ("jpg/Canon_40D.jpg",           "NoGPS_Canon_40D.jpg"),
    ("jpg/Nikon_COOLPIX_P1.jpg",    "NoGPS_Nikon_CoolPix.jpg"),
    ("jpg/Sony_HDR-HC3.jpg",        "NoGPS_Sony_HDR.jpg"),
    ("jpg/Fujifilm_FinePix_E500.jpg","NoGPS_Fujifilm.jpg"),
]

BASE_URL = "https://raw.githubusercontent.com/ianare/exif-samples/master/"

def download(url: str, dest: str) -> bool:
    try:
        r = requests.get(url, timeout=20)
        if r.status_code == 200:
            with open(dest, "wb") as f:
                f.write(r.content)
            return True
        else:
            print(f"  HTTP {r.status_code} — skipping")
            return False
    except Exception as e:
        print(f"  Error: {e}")
        return False

if __name__ == "__main__":
    print(f"Downloading sample images to: {SAMPLE_DIR}\n")
    ok = 0
    for path, filename in SAMPLES:
        url = BASE_URL + path
        dest = os.path.join(SAMPLE_DIR, filename)
        if os.path.exists(dest):
            print(f"  [exists] {filename}")
            ok += 1
            continue
        print(f"  Downloading {filename} ...")
        if download(url, dest):
            print(f"  [OK] {filename}")
            ok += 1
        else:
            print(f"  [FAIL] {filename}")

    print(f"\nDone: {ok}/{len(SAMPLES)} files in sample_data/")
    print("\nFiles with GPS (use these for the live GPS demo):")
    for _, fname in SAMPLES:
        if "GPS" in fname:
            fpath = os.path.join(SAMPLE_DIR, fname)
            if os.path.exists(fpath):
                print(f"  sample_data/{fname}")
