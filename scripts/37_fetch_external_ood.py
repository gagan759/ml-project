"""Final round, step 3 — a small set of REAL out-of-dataset photos, for evaluation only (never for fitting).

Sources (downloaded once into data/external_ood/, not committed, not redistributed):
- PlantVillage (Hughes & Salathé 2015, arXiv:1511.08060; github.com/spMohanty/PlantVillage-Dataset, raw/color):
  * other_crop          — leaves of 16 other crops / crop conditions (should be refused by both models);
  * tomato_unsupported  — tomato leaves with 7 diseases this project's dataset does not contain (should be refused);
  * tomato_known_shift  — tomato Early Blight, Late Blight and healthy leaves from another lab and camera
                          (a domain-shift check for the full model; Healthy also for the focused model).
- Lorem Picsum (Unsplash photos, Unsplash licence): everyday           — unrelated photographs (should be refused).

Files are picked deterministically by position in each folder's sorted listing (evenly spaced), never by looking
at them or at a model's output. Writes data/external_ood/manifest.csv (set, true key if any, source URL).
"""
import csv
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "external_ood"
API = "https://api.github.com/repos/spMohanty/PlantVillage-Dataset/contents/raw/color/"
RAW = "https://raw.githubusercontent.com/spMohanty/PlantVillage-Dataset/master/raw/color/"
UA = {"User-Agent": "tomato-care-evaluation"}
OTHER = ["Apple___healthy", "Apple___Apple_scab", "Blueberry___healthy", "Cherry_(including_sour)___healthy",
         "Corn_(maize)___healthy", "Corn_(maize)___Common_rust_", "Grape___healthy", "Grape___Black_rot",
         "Peach___healthy", "Pepper,_bell___healthy", "Potato___healthy", "Potato___Early_blight",
         "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy"]
UNSUPPORTED = ["Tomato___Bacterial_spot", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
               "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
               "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus"]
KNOWN = {"Tomato___Early_blight": "early_blight", "Tomato___Late_blight": "late_blight", "Tomato___healthy": "healthy"}
PER = {"other_crop": 6, "tomato_unsupported": 8, "tomato_known_shift": 12}
PICSUM_IDS = [int(i) for i in np.linspace(10, 1000, 70).round()]


def get(url: str, timeout: int = 60) -> bytes:
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - network: retry, then give up on this file
            if attempt == 2:
                raise exc
            time.sleep(2)


def listing(folder: str) -> list[str]:
    """Folder listing via the GitHub CLI when available (authenticated: no 60-requests/hour limit)."""
    try:
        out = subprocess.run(["gh", "api", API.split("api.github.com/")[1] + urllib.request.quote(folder)],
                             capture_output=True, text=True, check=True, timeout=120).stdout
        items = json.loads(out)
    except (OSError, subprocess.SubprocessError):
        items = json.loads(get(API + urllib.request.quote(folder)))
    return sorted(i["name"] for i in items if i["name"].lower().endswith((".jpg", ".jpeg", ".png")))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    plan = [("other_crop", f, None) for f in OTHER] + [("tomato_unsupported", f, None) for f in UNSUPPORTED] + \
           [("tomato_known_shift", f, k) for f, k in KNOWN.items()]
    for set_name, folder, key in plan:
        names = listing(folder)
        pick = [names[i] for i in np.linspace(0, len(names) - 1, PER[set_name]).round().astype(int)]
        for name in pick:
            dst = OUT / set_name / f"{folder.split('___')[0][:12]}__{folder.split('___')[-1][:24]}__{name}".replace(" ", "_")
            dst.parent.mkdir(parents=True, exist_ok=True)
            url = RAW + urllib.request.quote(f"{folder}/{name}")
            if not dst.exists():
                try:
                    dst.write_bytes(get(url))
                except Exception as exc:  # noqa: BLE001
                    print("skip", url, exc)
                    continue
            rows.append({"set": set_name, "file": str(dst.relative_to(ROOT)).replace("\\", "/"), "true_key": key or "",
                         "source": url, "folder": folder})
        print(f"{set_name:20s} {folder:45s} {len(pick)} files", flush=True)
    for pid in PICSUM_IDS:
        dst = OUT / "everyday" / f"picsum_{pid}.jpg"
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://picsum.photos/id/{pid}/640/480"
        if not dst.exists():
            try:
                dst.write_bytes(get(url))
            except Exception as exc:  # noqa: BLE001 - some ids do not exist
                print("skip", url, exc)
                continue
        rows.append({"set": "everyday", "file": str(dst.relative_to(ROOT)).replace("\\", "/"), "true_key": "",
                     "source": url, "folder": "picsum"})
    with open(OUT / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["set", "file", "true_key", "source", "folder"])
        w.writeheader()
        w.writerows(rows)
    print({s: sum(r["set"] == s for r in rows) for s in ("other_crop", "tomato_unsupported", "tomato_known_shift",
                                                           "everyday")})


if __name__ == "__main__":
    sys.exit(main())
