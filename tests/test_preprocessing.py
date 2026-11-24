"""The upload path must reproduce how the dataset was made, and must leave dataset images untouched."""
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from tomatoleaf.config import DATASET_DIR
from tomatoleaf.preprocessing import (
    P3_TO_SRGB,
    SRGB_TO_P3,
    canonicalize,
    canonicalize_legacy,
    convert_primaries,
    load_canonical_array,
    open_image,
    prepare_upload,
    strip_jpeg_metadata,
    to_dataset_colour,
)

HAS_DATA = DATASET_DIR.exists()


def jpeg(img: Image.Image, **kw) -> bytes:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=95, **kw)
    return buf.getvalue()


@pytest.fixture(scope="module")
def dataset_files():
    if not HAS_DATA:
        pytest.skip("dataset not available")
    import pandas as pd

    from tomatoleaf.config import DATA_DIR
    idx = pd.read_csv(DATA_DIR / "index.csv")
    return idx.sample(60, random_state=0)


def test_srgb_p3_matrices_preserve_neutrals_and_invert():
    grey = np.array([[[0, 0, 0], [64, 64, 64], [128, 128, 128], [255, 255, 255]]], dtype=np.uint8)
    assert np.array_equal(convert_primaries(grey, SRGB_TO_P3), grey)
    np.testing.assert_allclose(SRGB_TO_P3 @ P3_TO_SRGB, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(SRGB_TO_P3.sum(axis=1), 1.0, atol=1e-6)  # white maps to white
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    back = convert_primaries(convert_primaries(rgb, SRGB_TO_P3), P3_TO_SRGB)
    err = np.abs(back.astype(int) - rgb)
    # sRGB fits inside P3; 8-bit rounding of the intermediate costs a few levels only in dark saturated channels
    assert err.mean() < 0.5 and err.max() <= 6


def test_saturated_srgb_green_is_less_saturated_in_p3():
    p3 = convert_primaries(np.array([[[0, 255, 0]]], dtype=np.uint8), SRGB_TO_P3)[0, 0]
    assert p3[0] > 50 and p3[1] > 240 and p3[2] > 20


def test_dataset_images_unchanged(dataset_files):
    for rel in dataset_files.path:
        raw = np.asarray(open_image(DATASET_DIR / rel, manage_colour=False))
        assert np.array_equal(load_canonical_array(DATASET_DIR / rel), raw)


def test_dataset_original_unchanged_by_upload_path(dataset_files):
    originals = dataset_files[dataset_files.is_original].path[:15]
    for rel in originals:
        data = (DATASET_DIR / rel).read_bytes()
        rgb, info = prepare_upload(data)
        assert np.array_equal(rgb, load_canonical_array(DATASET_DIR / rel))
        assert info["colour"] == "Display P3 (unchanged)" and info["geometry"] == "already 256x256"


def test_three_by_four_photo_is_squashed_whole_not_cropped():
    # left and right thirds coloured differently: a centre crop would lose the top/bottom, a squash keeps everything
    img = Image.new("RGB", (300, 400), (255, 255, 255))
    img.paste((255, 0, 0), (0, 0, 300, 40))  # red band at the top (10% of the height)
    out = np.asarray(canonicalize(img, emulate_jpeg=False))
    assert out.shape == (256, 256, 3)
    assert out[5, 128, 0] > 200 and out[5, 128, 1] < 60  # still there after resizing the whole frame
    legacy = np.asarray(canonicalize_legacy(img))
    assert legacy[5, 128, 1] > 200  # the old centre crop cut the band off


def test_wide_photo_is_cropped_to_four_by_three():
    img = Image.new("RGB", (1600, 900), (0, 0, 255))
    img.paste((0, 255, 0), (0, 0, 200, 900))  # 12.5% wide band on the left: outside a centred 4:3 crop (1200 px)
    out = np.asarray(canonicalize(img, emulate_jpeg=False))
    assert out.shape == (256, 256, 3)
    assert out[:, :4, 1].mean() < 30  # band removed


def test_upload_reports_colour_and_geometry():
    img = Image.new("RGB", (3000, 4000), (90, 150, 70))
    rgb, info = prepare_upload(jpeg(img))
    assert rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8
    assert info["width"] == 3000 and info["height"] == 4000
    assert "assumed sRGB" in info["colour"] and "whole frame" in info["geometry"]


def test_p3_tagged_upload_is_not_converted():
    if not HAS_DATA:
        pytest.skip("dataset not available (needs a Display P3 profile)")
    import pandas as pd

    from tomatoleaf.config import DATA_DIR
    rel = pd.read_csv(DATA_DIR / "index.csv").query("is_original").path.iloc[0]
    icc = Image.open(DATASET_DIR / rel).info["icc_profile"]
    img = Image.new("RGB", (64, 64), (90, 150, 70))
    img.info["icc_profile"] = icc
    out, label = to_dataset_colour(img)
    assert label == "Display P3 (unchanged)" and np.asarray(out)[0, 0].tolist() == [90, 150, 70]


def test_exif_orientation_is_applied():
    img = Image.new("RGB", (400, 300), (0, 0, 0))
    img.paste((255, 255, 255), (0, 0, 100, 300))  # white on the left
    exif = Image.Exif()
    exif[0x0112] = 6  # rotate 90 degrees clockwise on display
    rgb, info = prepare_upload(jpeg(img, exif=exif))
    assert (info["width"], info["height"]) == (300, 400)
    assert rgb[:20].mean() > 200  # the white left edge is now at the top


def test_transparent_png_is_flattened_on_white():
    img = Image.new("RGBA", (300, 300), (0, 0, 0, 0))
    buf = BytesIO()
    img.save(buf, format="PNG")
    rgb, _ = prepare_upload(buf.getvalue())
    assert rgb.mean() > 240


def test_sixteen_bit_png_is_supported():
    arr = (np.linspace(0, 65535, 256 * 256).reshape(256, 256)).astype(np.uint16)
    buf = BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    rgb, _ = prepare_upload(buf.getvalue())
    assert rgb.shape == (256, 256, 3) and rgb.max() > 200


def test_strip_jpeg_metadata_is_lossless_and_keeps_the_colour_profile():
    """Library images are dataset files minus metadata: same decoded pixels, Display P3 profile kept."""
    from PIL import ImageCms

    p3 = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()  # any valid ICC payload
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
    raw = jpeg(img, icc_profile=p3)
    app4 = bytes([0xFF, 0xE4])
    junk = app4 + (2 + 1000).to_bytes(2, "big") + bytes(1000)  # an APP4 debug segment, like the phone's
    raw = raw[:2] + junk + raw[2:]
    out = strip_jpeg_metadata(raw)
    assert len(out) < len(raw) - 1000
    assert app4 not in out[:200]
    assert np.array_equal(np.asarray(Image.open(BytesIO(out))), np.asarray(Image.open(BytesIO(raw))))
    assert Image.open(BytesIO(out)).info.get("icc_profile") == p3
    with pytest.raises(ValueError):
        strip_jpeg_metadata(b"not a jpeg")
