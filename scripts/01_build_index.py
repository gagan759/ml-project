"""Phase 1a — index every image in Variant-a and quantify the leakage in the official split.

Outputs: data/index.csv, experiments/results/leakage_official_split.json
"""
import hashlib
import struct
from collections import Counter

from PIL import Image

from tomatoleaf.config import DATA_DIR, DATASET_DIR, RESULTS_DIR
from tomatoleaf.data import build_index, leakage_report
from tomatoleaf.utils import save_json

# IJG standard luminance quantisation table in zig-zag order (as stored in DQT segments)
STD_LUMA = [16, 11, 12, 14, 12, 10, 16, 14, 13, 14, 18, 17, 16, 19, 24, 40, 26, 24, 22, 22, 24, 49, 35, 37, 29, 40,
            58, 51, 61, 60, 57, 51, 56, 55, 64, 72, 92, 78, 64, 68, 87, 69, 55, 56, 80, 109, 81, 87, 95, 98, 103, 104,
            103, 62, 77, 113, 121, 112, 100, 120, 92, 101, 103, 99]


def jpeg_header_info(data: bytes) -> dict:
    """Metadata bytes (APPn/COM: EXIF, ICC, camera debug) and estimated IJG quality from the luminance table."""
    i, meta, quality = 2, 0, None
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        if marker == 0xDA:
            break
        length = struct.unpack(">H", data[i + 2:i + 4])[0]
        segment = data[i + 4:i + 2 + length]
        if 0xE0 <= marker <= 0xEF or marker == 0xFE:
            meta += 2 + length
        if marker == 0xDB and quality is None and segment[0] == 0:
            scale = sum(q * 100.0 / s for q, s in zip(segment[1:65], STD_LUMA)) / 64
            quality = round(100 - scale / 2) if scale <= 100 else round(5000 / scale)
        i += 2 + length
    return {"metadata_bytes": meta, "jpeg_quality": quality}


def fully_decodes(path) -> bool:
    try:
        with Image.open(path) as img:
            img.load()
        return True
    except Exception:
        return False


def main():
    index = build_index()
    raw = [(DATASET_DIR / p).read_bytes() for p in index.path]
    info = [jpeg_header_info(b) for b in raw]
    index["file_bytes"] = [len(b) for b in raw]
    index["metadata_bytes"] = [x["metadata_bytes"] for x in info]
    index["jpeg_quality"] = [x["jpeg_quality"] for x in info]
    index["md5"] = [hashlib.md5(b).hexdigest() for b in raw]
    index["decodes"] = [fully_decodes(DATASET_DIR / p) for p in index.path]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    index.to_csv(DATA_DIR / "index.csv", index=False)

    report = leakage_report(index)
    report["corrupt_files"] = int((~index.decodes).sum())
    report["byte_identical_duplicate_files"] = int(index.md5.duplicated(keep=False).sum())
    report["jpeg_quality_counts"] = {str(k): v for k, v in Counter(index.jpeg_quality).items()}
    by_type = index.groupby("is_original")[["file_bytes", "metadata_bytes"]].mean()
    report["mean_bytes"] = {
        "original": {"file": float(by_type.loc[True, "file_bytes"]), "metadata": float(by_type.loc[True, "metadata_bytes"])},
        "copy": {"file": float(by_type.loc[False, "file_bytes"]), "metadata": float(by_type.loc[False, "metadata_bytes"])},
    }
    save_json(report, RESULTS_DIR / "leakage_official_split.json")

    print(f"Indexed {report['total_images']} images from {report['unique_photos']} unique photos")
    print(f"Augmented families: {report['augmented_families']} ({report['augmented_copies']} copies); "
          f"crossing official splits: {report['families_crossing_splits']}")
    for split in ("val", "test"):
        n, leak = report[f"{split}_images"], report[f"{split}_images_with_train_sibling"]
        print(f"{split}: {leak}/{n} images ({100 * leak / n:.1f}%) have a sibling of the same photo in train")
    print("Mean bytes (file / metadata):", report["mean_bytes"])
    print(f"Corrupt files: {report['corrupt_files']} | byte-identical duplicates: {report['byte_identical_duplicate_files']} "
          f"| JPEG quality estimates: {report['jpeg_quality_counts']}")


if __name__ == "__main__":
    main()
