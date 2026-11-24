"""End-to-end check of the Streamlit app in a real browser (Playwright + Edge/Chromium).

Every page at desktop and phone width in the light and the dark theme: no Python exception on the page, no
horizontal scrolling, no browser console errors. On Diagnose: an example photo, a full-size phone-style JPEG upload,
an invalid file, the classical ML model and the two-model comparison. Screenshots go to docs/screenshots/.

    streamlit run streamlit_app.py --server.port 8510      # in another terminal
    python scripts/30_streamlit_check.py --url http://localhost:8510
    python scripts/30_streamlit_check.py --url https://<your-app>.streamlit.app --no-shots   # the deployed app
"""
import argparse
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "docs" / "screenshots"
PAGES = {"home": "", "diagnose": "diagnose", "lab": "lab", "library": "library", "method": "how-it-works"}
VIEWPORTS = {"desktop": {"width": 1440, "height": 900}, "phone": {"width": 390, "height": 844}}
# screenshots kept in the repository (README and docs): (page, viewport, scheme)
SHOTS_WANTED = {("home", "desktop", "light"), ("home", "desktop", "dark"), ("home", "phone", "light"),
                ("lab", "desktop", "light"), ("library", "desktop", "light"), ("method", "desktop", "light"),
                ("diagnose", "phone", "light")}


def settle(page, timeout=180):
    """Wait until Streamlit has finished running the script (status widget gone, no spinner)."""
    page.wait_for_selector('[data-testid="stAppViewContainer"]', timeout=timeout * 1000)
    t0 = time.time()
    quiet = 0
    while time.time() - t0 < timeout:
        busy = page.locator('[data-testid="stStatusWidget"], [data-testid="stSpinner"]').count()
        quiet = quiet + 1 if busy == 0 else 0
        if quiet >= 3:
            return
        page.wait_for_timeout(400)
    raise TimeoutError("app did not settle")


def problems(page) -> list[str]:
    out = []
    if page.locator('[data-testid="stException"]').count():
        out.append("exception: " + page.locator('[data-testid="stException"]').first.inner_text()[:300])
    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
    if overflow > 1:
        out.append(f"horizontal overflow {overflow}px")
    return out


def segment(page, label: str):
    """A segmented-control option (Streamlit renders them as radio buttons in a button group)."""
    return page.locator('[data-testid="stButtonGroup"] button[role="radio"]', has_text=label).first


def phone_photo() -> bytes:
    """A held-out example turned into a full-size 3:4 phone-style JPEG (exercises the upload conversion)."""
    img = Image.open(ROOT / "frontend" / "public" / "test-samples" / "leaf_miner.jpg").convert("RGB").resize((1536, 2048))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8510")
    parser.add_argument("--no-shots", action="store_true")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    if base.endswith(".streamlit.app"):  # Community Cloud serves the app itself in an iframe at /~/+/
        base += "/~/+"
    report, failures = {"url": base, "pages": [], "diagnose": {}}, []
    SHOTS.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:
            browser = p.chromium.launch()
        for scheme in ("light", "dark"):
            for vp_name, vp in VIEWPORTS.items():
                ctx = browser.new_context(viewport=vp, color_scheme=scheme, device_scale_factor=1)
                page = ctx.new_page()
                page.set_default_navigation_timeout(120000)  # a cold Community Cloud app can be slow
                errors = []
                # Streamlit probes <page>/_stcore/... on deep links before falling back to the root: harmless 404s
                page.on("console", lambda m, errors=errors: errors.append(m.text) if m.type == "error"
                        and "/_stcore/" not in (m.location or {}).get("url", "") else None)
                for name, path in PAGES.items():
                    page.goto(f"{base}/{path}", wait_until="domcontentloaded")
                    settle(page)
                    page.wait_for_timeout(600)
                    issues = problems(page)
                    report["pages"].append({"page": name, "viewport": vp_name, "scheme": scheme, "issues": issues})
                    if issues:
                        failures.append(f"{name}/{vp_name}/{scheme}: {issues}")
                    if not args.no_shots and (name, vp_name, scheme) in SHOTS_WANTED:
                        page.screenshot(path=str(SHOTS / f"streamlit-{name}-{vp_name}-{scheme}.png"))
                    print(f"{scheme:5s} {vp_name:7s} {name:9s} {'OK' if not issues else issues}", flush=True)
                real = [e for e in errors if "favicon" not in e.lower() and "fonts.g" not in e.lower()]
                if real:
                    failures.append(f"console errors ({scheme}/{vp_name}): {real[:3]}")
                ctx.close()

        # ---- diagnose flows (desktop, light)
        ctx = browser.new_context(viewport=VIEWPORTS["desktop"], color_scheme="light")
        page = ctx.new_page()
        page.set_default_navigation_timeout(120000)
        page.goto(f"{base}/diagnose", wait_until="domcontentloaded")
        settle(page)
        segment(page, "Examples").click()
        settle(page)
        page.get_by_role("button", name="Leaf miner", exact=True).first.click()
        settle(page)
        page.wait_for_selector(".tc-status", timeout=120000)
        status = page.locator(".tc-status").first.inner_text()
        report["diagnose"]["example"] = status
        print("example ->", status.replace("\n", " | "))
        if not args.no_shots:
            page.screenshot(path=str(SHOTS / "streamlit-diagnose-result.png"))
        page.get_by_role("tab", name="Where it looked").click()
        page.wait_for_timeout(800)
        if page.locator('[data-testid="stImage"]').count() < 2:
            failures.append("Grad-CAM images missing")
        # two-model comparison
        page.get_by_text("Ask the other model too").click()
        settle(page)
        page.wait_for_selector('[data-testid="stDataFrame"]', timeout=180000)
        report["diagnose"]["compare"] = "shown"
        # classical ML model
        deep = page.locator(".tc-status").first.inner_text()
        dep = json.loads((ROOT / "experiments" / "deployment.json").read_text(encoding="utf-8"))
        ml = next(e for e in dep["models"] if e.get("type") == "sklearn")
        dl = next(e for e in dep["models"] if e["key"] == dep["default_model"])
        segment(page, ml["model_name"]).click()
        page.wait_for_selector(f"text={ml['model_name']} · classical ML", timeout=300000)
        for _ in range(60):  # a slow server may still show the previous answer for a moment
            text = page.locator(".tc-status").first.inner_text() if page.locator(".tc-status").count() else ""
            if text and text != deep:
                break
            page.wait_for_timeout(2000)
        report["diagnose"]["example_classical"] = text
        print("classical ->", report["diagnose"]["example_classical"].replace("\n", " | "))
        segment(page, dl["model_name"]).click()
        settle(page)
        # upload a phone-size photo
        segment(page, "Upload").click()
        settle(page)
        page.locator('input[type="file"]').set_input_files(
            {"name": "phone.jpg", "mimeType": "image/jpeg", "buffer": phone_photo()})
        settle(page)
        page.wait_for_selector(".tc-status", timeout=120000)
        report["diagnose"]["upload"] = page.locator(".tc-status").first.inner_text()
        print("upload ->", report["diagnose"]["upload"].replace("\n", " | "))
        # invalid file
        page.locator('input[type="file"]').set_input_files(
            {"name": "not-a-photo.jpg", "mimeType": "image/jpeg", "buffer": b"this is not an image"})
        try:
            page.wait_for_selector("text=could not be read as an image", timeout=120000)
        except Exception:
            pass
        err = page.locator('[data-testid="stAlertContentError"], [data-testid="stAlert"]').first
        report["diagnose"]["invalid"] = err.inner_text() if err.count() else None
        print("invalid ->", report["diagnose"]["invalid"])
        if not report["diagnose"]["invalid"] or "could not be read" not in report["diagnose"]["invalid"]:
            failures.append("invalid file was not rejected with a message")
        if problems(page):
            failures.append(f"diagnose flow: {problems(page)}")
        ctx.close()
        browser.close()
    report["failures"] = failures
    (ROOT / "experiments" / "results" / "streamlit_app_check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\nALL CHECKS PASSED" if not failures else "\nFAILURES:\n- " + "\n- ".join(failures))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
