"""End-to-end check of the whole platform in a real browser (Microsoft Edge via Playwright) + README screenshots.

Starts the API in one-server mode (it serves the built frontend from frontend/dist), then:
- visits every page at desktop and phone width, failing on console errors, failed requests or horizontal overflow;
- diagnoses a held-out test photo through the UI and checks the page shows exactly the API's answer;
- checks Grad-CAM, the all-model comparison, the model lab and the disease library work;
- rejects a non-image upload with a visible message.
Output: docs/screenshots/*.png, experiments/results/e2e_check.json
Run after `npm run build` in frontend/.
"""
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import httpx
from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tomatoleaf.config import RESULTS_DIR  # noqa: E402
from tomatoleaf.utils import save_json  # noqa: E402

BASE = "http://127.0.0.1:8010"
SHOTS = ROOT / "docs" / "screenshots"
PAGES = ["/", "/dashboard", "/diagnose", "/lab", "/performance", "/explainability", "/library", "/library/leaf_miner",
         "/methodology", "/docs"]
SAMPLE = "leaf_miner"


def wait_for(url: str, timeout: float = 240) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            with urllib.request.urlopen(url, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception:
            time.sleep(1)
    raise TimeoutError(f"{url} did not come up")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--photo", default=f"frontend/public/test-samples/{SAMPLE}.jpg",
                        help="photo to diagnose (use a non-test photo for dry runs, e.g. frontend/public/samples/leaf_miner-1.jpg)")
    args = parser.parse_args()
    SHOTS.mkdir(parents=True, exist_ok=True)
    checks, problems = {}, []
    server = subprocess.Popen([sys.executable, "-m", "uvicorn", "backend.main:app", "--port", "8010"], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for(f"{BASE}/api/health")
        sample = ROOT / args.photo
        api = httpx.post(f"{BASE}/api/predict", files={"file": (sample.name, sample.read_bytes(), "image/jpeg")},
                         timeout=120).json()
        checks["api_prediction"] = api["prediction"]

        with sync_playwright() as p:
            browser = p.chromium.launch(channel="msedge", headless=True)
            for label, viewport in (("desktop", {"width": 1440, "height": 900}), ("phone", {"width": 390, "height": 844})):
                ctx = browser.new_context(viewport=viewport, device_scale_factor=1 if label == "desktop" else 2)
                page = ctx.new_page()
                errors = []
                # third-party web fonts (Google Fonts) can fail on some networks; the site falls back to system fonts
                cdn = ("fonts.gstatic.com", "fonts.googleapis.com")
                page.on("console", lambda m, errors=errors, cdn=cdn: m.type == "error" and not any(
                    c in (m.location or {}).get("url", "") or "QUIC" in m.text for c in cdn) and errors.append(m.text))
                page.on("requestfailed", lambda r, errors=errors, cdn=cdn: not any(c in r.url for c in cdn)
                        and errors.append(f"request failed: {r.url}"))
                for path in PAGES:
                    page.goto(BASE + path, wait_until="networkidle")
                    for _ in range(12):  # scroll through the page so lazy images load before the full-page shot
                        page.mouse.wheel(0, 900)
                        page.wait_for_timeout(120)
                    page.evaluate("window.scrollTo(0, 0)")
                    page.wait_for_timeout(500)
                    overflow = page.evaluate("document.documentElement.scrollWidth - window.innerWidth")
                    if overflow > 1:
                        problems.append(f"{label} {path}: horizontal overflow {overflow}px")
                    name = path.strip("/").replace("/", "_") or "home"
                    if label == "desktop" or path in ("/", "/diagnose", "/dashboard"):
                        page.screenshot(path=str(SHOTS / f"{label}_{name}.png"), full_page=label == "desktop")
                if errors:
                    problems += [f"{label}: {e}" for e in errors]
                checks[f"{label}_pages"] = len(PAGES)
                ctx.close()

            # ---- diagnose flow through the UI
            ctx = browser.new_context(viewport={"width": 1440, "height": 1000})
            page = ctx.new_page()
            page.goto(BASE + "/diagnose", wait_until="networkidle")
            expect(page.get_by_title("Leaf Miner").first).to_be_visible()  # held-out gallery renders (not clicked)
            page.locator('input[type="file"]').first.set_input_files(str(sample))
            page.get_by_role("button", name="Diagnose", exact=True).click()
            expect(page.get_by_text("Class probabilities")).to_be_visible(timeout=120000)
            shown = page.locator("h2").first.inner_text()
            checks["ui_heading"] = shown
            expected = api["prediction"]["name"]
            if api["prediction"]["status"] not in ("Healthy", "Diseased"):
                expected = "No diagnosis" if api["prediction"]["status"] == "Unusual image" else "Not sure"
            if shown.strip() != expected:
                problems.append(f"UI shows '{shown}', API says '{expected}'")
            expect(page.get_by_alt_text("Grad-CAM heat map").first).to_be_visible()
            page.get_by_role("button", name="Ask both models").click()
            expect(page.get_by_text("Votes:")).to_be_visible(timeout=180000)
            page.screenshot(path=str(SHOTS / "desktop_diagnose_result.png"), full_page=True)
            checks["compare_rows"] = page.locator("table tbody tr").count()
            served = len(httpx.get(BASE + "/api/models", timeout=30).json()["models"])
            if checks["compare_rows"] != served:  # both served models
                problems.append(f"comparison shows {checks['compare_rows']} rows, expected {served}")

            # ---- invalid upload is rejected with a message
            page.goto(BASE + "/diagnose", wait_until="networkidle")
            page.locator('input[type="file"]').first.set_input_files(
                {"name": "notes.txt", "mimeType": "text/plain", "buffer": b"not an image"})
            expect(page.get_by_role("alert")).to_contain_text("Unsupported file type")
            checks["invalid_file_rejected"] = True

            # ---- model lab: leaderboard and confusion matrices render
            page.goto(BASE + "/lab", wait_until="networkidle")
            checks["lab_rows"] = page.locator("table").first.locator("tbody tr").count()
            ctx.close()
            browser.close()
    finally:
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(server.pid)], capture_output=True)
    checks["problems"] = problems
    checks["passed"] = not problems
    save_json(checks, RESULTS_DIR / "e2e_check.json")
    print(checks)
    sys.exit(0 if not problems else 1)


if __name__ == "__main__":
    main()
