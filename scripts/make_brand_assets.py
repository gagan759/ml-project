"""Tomato Care brand assets: favicons, app icons, wordmarks and the social card, rendered from two SVG sources.

Sources (edit these, then re-run):
  assets/brand/mark.svg        logo mark: tomato + pulse line on a green tile (app icons, logos)
  assets/brand/mark-small.svg  bolder variant that stays readable at 16-32 px (favicons)

Every raster file is rendered by a headless browser (Playwright: Microsoft Edge if installed, else Playwright's
Chromium) so it matches the SVG exactly; favicon.ico is assembled from separately rendered 16/32/48 px images.

    python scripts/make_brand_assets.py
"""
import json
import re
import shutil
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
BRAND = ROOT / "assets" / "brand"
PUBLIC = ROOT / "frontend" / "public"
FONTS = "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Sora:wght@600;700&display=swap"
INK = {"light": "#0e1a14", "dark": "#e9f0ec"}
CARE = {"light": "#15834e", "dark": "#46be7d"}
TAGLINE = "Tomato leaf disease diagnosis from a single photo"


def sized(svg: str, w: int, h: int | None = None) -> str:
    return svg.replace("<svg ", f'<svg width="{w}" height="{h or w}" ', 1)


def full_bleed(svg: str) -> str:
    """No rounded corners: for platforms that apply their own mask (iOS home screen)."""
    return re.sub(r'(<rect width="64" height="64") rx="\d+"', r"\1", svg, count=1)


def maskable(svg: str) -> str:
    """Full-bleed tile with the artwork shrunk into the central 80% safe zone (Android adaptive icons)."""
    svg = full_bleed(svg)
    head, rest = svg.split('<rect width="64" height="64"', 1)
    rect_end = rest.index("/>") + 2
    rect, body = '<rect width="64" height="64"' + rest[:rect_end], rest[rect_end:]
    body, tail = body.rsplit("</svg>", 1)
    return f'{head}{rect}<g transform="translate(6.4 6.4) scale(.8)">{body}</g></svg>{tail}'


def page_html(body: str, fonts: bool = False) -> str:
    link = f'<link rel="stylesheet" href="{FONTS}">' if fonts else ""
    return (f"<!doctype html><html><head><meta charset='utf-8'>{link}<style>html,body{{margin:0;background:transparent}}"
            f"</style></head><body>{body}</body></html>")


def wordmark(svg: str, theme: str) -> str:
    return page_html(
        f"<div id='shot' style='display:inline-flex;align-items:center;gap:14px;padding:6px 10px 6px 6px;"
        f"font-family:Sora,sans-serif;font-weight:700;font-size:34px;letter-spacing:-0.02em;line-height:1'>"
        f"{sized(svg, 52)}<span><span style='color:{INK[theme]}'>Tomato</span> "
        f"<span style='color:{CARE[theme]}'>Care</span></span></div>", fonts=True)


def social_card(svg: str) -> str:
    chips = "".join(
        f"<span style='border:1px solid rgba(255,255,255,.18);background:rgba(255,255,255,.06);border-radius:999px;"
        f"padding:10px 20px'>{c}</span>"
        for c in ("8 leaf conditions", "Calibrated confidence", "Grad-CAM", "Unusual-image check"))
    pulse = ("<svg width='1200' height='630' viewBox='0 0 1200 630' style='position:absolute;inset:0'>"
             "<path d='M-20 420h520l46-104 74 186 62-138 38 52h500' fill='none' stroke='#7be3a6' stroke-opacity='.10' "
             "stroke-width='14' stroke-linecap='round' stroke-linejoin='round'/></svg>")
    return page_html(
        "<div id='shot' style='width:1200px;height:630px;position:relative;overflow:hidden;color:#fff;"
        "font-family:Inter,sans-serif;background:radial-gradient(900px 520px at 88% 12%,rgba(70,190,125,.35),"
        "transparent 62%),linear-gradient(135deg,#0b3b2c 0%,#041a11 100%)'>"
        f"{pulse}<div style='position:absolute;left:84px;top:84px'>{sized(svg, 128)}"
        "<div style='font-family:Sora,sans-serif;font-weight:700;font-size:100px;letter-spacing:-0.035em;"
        "margin-top:30px;line-height:1'>Tomato <span style='color:#7be3a6'>Care</span></div>"
        f"<div style='font-size:36px;color:#c5d1ca;margin-top:22px;max-width:900px;line-height:1.3'>{TAGLINE} &mdash; "
        "with an honest &ldquo;not sure&rdquo; and a visual explanation.</div></div>"
        f"<div style='position:absolute;left:84px;bottom:70px;display:flex;gap:14px;font-size:22px;"
        f"color:#e9f0ec'>{chips}</div></div>", fonts=True)


def main():
    mark = (BRAND / "mark.svg").read_text(encoding="utf-8")
    small = (BRAND / "mark-small.svg").read_text(encoding="utf-8")
    PUBLIC.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        try:
            browser = p.chromium.launch(channel="msedge")
        except Exception:
            browser = p.chromium.launch()

        def shoot(html: str, out: Path, w: int, h: int, scale: float = 1, element: bool = False):
            page = browser.new_page(viewport={"width": w, "height": h}, device_scale_factor=scale)
            page.set_content(html, wait_until="networkidle")
            page.evaluate("document.fonts.ready")
            target = page.locator("#shot") if element else page
            target.screenshot(path=str(out), omit_background=True)
            page.close()
            print(f"  {out.relative_to(ROOT)}")

        def icon(svg: str, out: Path, size: int):
            shoot(page_html(sized(svg, size)), out, size, size)

        tmp = BRAND / "_ico"
        tmp.mkdir(exist_ok=True)
        for s in (16, 32, 48):
            icon(small, tmp / f"{s}.png", s)
        icon(full_bleed(mark), PUBLIC / "apple-touch-icon.png", 180)
        icon(mark, PUBLIC / "icon-192.png", 192)
        icon(mark, PUBLIC / "icon-512.png", 512)
        icon(maskable(mark), PUBLIC / "icon-maskable-512.png", 512)
        icon(mark, BRAND / "mark-256.png", 256)
        icon(small, BRAND / "mark-small-64.png", 64)
        shoot(wordmark(mark, "light"), BRAND / "logo-light.png", 600, 120, scale=2, element=True)
        shoot(wordmark(mark, "dark"), BRAND / "logo-dark.png", 600, 120, scale=2, element=True)
        shoot(social_card(mark), PUBLIC / "og-image.png", 1200, 630, element=True)
        browser.close()

    imgs = [Image.open(tmp / f"{s}.png").convert("RGBA") for s in (16, 32, 48)]
    imgs[2].save(PUBLIC / "favicon.ico", format="ICO", sizes=[(16, 16), (32, 32), (48, 48)], append_images=imgs[:2])
    shutil.rmtree(tmp)
    shutil.copyfile(BRAND / "mark-small.svg", PUBLIC / "favicon.svg")
    shutil.copyfile(PUBLIC / "og-image.png", BRAND / "social-card.png")
    manifest = {
        "name": "Tomato Care", "short_name": "Tomato Care",
        "description": f"{TAGLINE}: calibrated confidence, Grad-CAM explanations and an unusual-image check.",
        "start_url": "/", "scope": "/", "display": "standalone",
        "background_color": "#f5f7f3", "theme_color": "#0b3b2c",
        "icons": [{"src": "/icon-192.png", "sizes": "192x192", "type": "image/png"},
                  {"src": "/icon-512.png", "sizes": "512x512", "type": "image/png"},
                  {"src": "/icon-maskable-512.png", "sizes": "512x512", "type": "image/png", "purpose": "maskable"}],
    }
    (PUBLIC / "site.webmanifest").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  frontend/public/favicon.ico, favicon.svg, site.webmanifest, assets/brand/social-card.png")


if __name__ == "__main__":
    main()
