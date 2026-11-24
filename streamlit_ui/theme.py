"""Look and feel: global CSS on top of the Streamlit theme (.streamlit/config.toml) and small HTML building blocks.

Colours that must work in both the light and the dark theme are translucent tints of the brand colours; blocks
with their own background (the hero) set their own text colour.
"""
import base64
import html

import streamlit as st

from . import core

# Icons: Lucide (ISC licence), inlined so they need no icon font.
_SVG = {
    "eco": '<path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z"/>'
           '<path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12"/>',
    "coronavirus": '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3"/>'
                   '<path d="M12 9v4"/><path d="M12 17h.01"/>',
    "help": '<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><path d="M12 17h.01"/>',
    "image_not_supported": '<line x1="2" x2="22" y1="2" y2="22"/><path d="M10.41 10.41a2 2 0 1 1-2.83-2.83"/>'
                           '<line x1="13.5" x2="6" y1="13.5" y2="21"/><line x1="18" x2="21" y1="12" y2="15"/>'
                           '<path d="M3.59 3.59A1.99 1.99 0 0 0 3 5v14a2 2 0 0 0 2 2h14c.55 0 1.052-.22 1.41-.59"/>'
                           '<path d="M21 15V5a2 2 0 0 0-2-2H9"/>',
    "tune": '<line x1="21" x2="14" y1="4" y2="4"/><line x1="10" x2="3" y1="4" y2="4"/><line x1="21" x2="12" y1="12" '
            'y2="12"/><line x1="8" x2="3" y1="12" y2="12"/><line x1="21" x2="16" y1="20" y2="20"/><line x1="12" x2="3" '
            'y1="20" y2="20"/><line x1="14" x2="14" y1="2" y2="6"/><line x1="8" x2="8" y1="10" y2="14"/>'
            '<line x1="16" x2="16" y1="18" y2="22"/>',
    "account_tree": '<rect x="16" y="16" width="6" height="6" rx="1"/><rect x="2" y="16" width="6" height="6" rx="1"/>'
                    '<rect x="9" y="2" width="6" height="6" rx="1"/><path d="M5 16v-3a1 1 0 0 1 1-1h12a1 1 0 0 1 1 1v3"/>'
                    '<path d="M12 12V8"/>',
    "groups": '<path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/>'
              '<path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
    "visibility": '<path d="M2.062 12.348a1 1 0 0 1 0-.696 10.75 10.75 0 0 1 19.876 0 1 1 0 0 1 0 .696 10.75 10.75 0 0 1-19.876 0"/>'
                  '<circle cx="12" cy="12" r="3"/>',
}


def icon(name: str, color: str = "#ffffff") -> str:
    """A white line icon as an <img> data URL (st.html drops inline <svg>)."""
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
           f'stroke-linecap="round" stroke-linejoin="round">{_SVG[name]}</svg>')
    return f'<img alt="" src="data:image/svg+xml;base64,{base64.b64encode(svg.encode()).decode()}">'


STATUS = {
    "Healthy": {"tone": "green", "icon": "eco", "title": "Healthy leaf",
                "text": "No disease or deficiency recognised."},
    "Diseased": {"tone": "red", "icon": "coronavirus", "title": "Condition detected",
                 "text": "See the condition below for symptoms and management."},
    "Uncertain": {"tone": "amber", "icon": "help", "title": "Not sure",
                  "text": "Confidence is below this model's threshold, so no diagnosis is given. Try a sharper photo "
                          "of a single leaf on plain paper, or ask an expert."},
    "Unusual image": {"tone": "violet", "icon": "image_not_supported", "title": "Unusual image",
                      "text": "This photo looks unlike every training photo (a different crop, scene or condition?), "
                              "so no diagnosis is given."},
}

CSS = """
<style>
:root { --tc-green:#15834e; --tc-green-2:#22a262; --tc-red:#e0472f; --tc-amber:#e08a0b; --tc-violet:#7c3aed; }
[data-testid="stMainBlockContainer"] { max-width: 1240px; padding-top: 4.9rem; padding-bottom: 3rem; }
h1, h2, h3 { letter-spacing: -0.02em; }
[data-testid="stImage"] img { border-radius: 14px; }
[data-testid="stMetric"] { padding: 0.9rem 1rem; }
[data-testid="stMetricValue"] { font-family: Sora, Inter, sans-serif; font-weight: 650; letter-spacing: -0.02em; }
[data-testid="stCaptionContainer"] { line-height: 1.45; }

/* hero */
.st-key-hero { position: relative; overflow: hidden; border-radius: 26px; padding: 2.6rem 2.6rem 2.2rem;
  background: radial-gradient(900px 420px at 88% -10%, rgba(70,190,125,.38), transparent 60%),
              radial-gradient(600px 300px at 0% 110%, rgba(240,84,63,.20), transparent 60%),
              linear-gradient(135deg, #0b3b2c 0%, #041a11 100%); }
.st-key-hero [data-testid="stMarkdownContainer"], .st-key-hero [data-testid="stCaptionContainer"] { color: #d5e2da; }
.st-key-hero button[kind="secondary"] { background: rgba(255,255,255,.06); color: #fff; border-color: rgba(255,255,255,.38); }
.st-key-hero button[kind="secondary"]:hover { background: rgba(255,255,255,.14); color: #fff; border-color: #7be3a6; }
.st-key-hero button[kind="secondary"] p, .st-key-hero button[kind="secondary"] span { color: #fff; }
.st-key-hero button[kind="primary"] { background: #22a262; border-color: #22a262; color: #fff; }
.st-key-hero button[kind="primary"]:hover { background: #1d8f56; border-color: #1d8f56; }
.tc-eyebrow { display:inline-flex; gap:.5rem; align-items:center; font-size:.78rem; font-weight:600; letter-spacing:.08em;
  text-transform:uppercase; color:#9fe3bd; background:rgba(255,255,255,.07); border:1px solid rgba(255,255,255,.14);
  padding:.35rem .8rem; border-radius:999px; }
.tc-hero-title { font-family:Sora, Inter, sans-serif; font-weight:700; font-size:clamp(2.1rem, 4.4vw, 3.4rem);
  line-height:1.06; letter-spacing:-0.035em; color:#fff; margin:1rem 0 .9rem; }
.tc-hero-title span { color:#7be3a6; }
.tc-hero-sub { color:#c5d6cc; font-size:1.08rem; line-height:1.6; max-width:38rem; margin:0; }
.tc-hero-figure { position:relative; }
.tc-hero-figure img { width:100%; border-radius:18px; box-shadow:0 30px 60px -20px rgba(0,0,0,.55); display:block; }
.tc-hero-badge { position:absolute; left:14px; bottom:14px; right:14px; display:flex; justify-content:space-between;
  gap:.6rem; align-items:center; background:rgba(4,26,17,.78); backdrop-filter:blur(8px); color:#fff;
  border:1px solid rgba(255,255,255,.14); border-radius:14px; padding:.65rem .85rem; font-size:.86rem; }
.tc-hero-badge b { font-family:Sora, Inter, sans-serif; font-size:1rem; }
.tc-hero-badge .p { color:#7be3a6; font-weight:700; font-family:Sora, Inter, sans-serif; font-size:1.15rem; }

/* section titles */
.tc-kicker { font-size:.76rem; font-weight:700; letter-spacing:.09em; text-transform:uppercase; color:var(--tc-green-2);
  margin:0 0 .25rem; }
.tc-section-title { font-family:Sora, Inter, sans-serif; font-weight:650; font-size:1.55rem; letter-spacing:-0.02em;
  margin:0 0 .3rem; }
.tc-section-sub { opacity:.75; margin:0 0 .4rem; max-width:52rem; line-height:1.55; }

/* status card */
.tc-status { border-radius:18px; padding:1.1rem 1.2rem; border:1px solid; display:flex; gap:1rem; align-items:flex-start; }
.tc-status .ic { flex:none; width:2.6rem; height:2.6rem; border-radius:12px; padding:.55rem; color:#fff; display:inline-flex; }
.tc-status .ic img, .tc-feature-icon img { width:100%; height:100%; display:block; }
.tc-status .lbl { font-size:.78rem; font-weight:700; letter-spacing:.07em; text-transform:uppercase; opacity:.85; }
.tc-status .name { font-family:Sora, Inter, sans-serif; font-weight:700; font-size:1.65rem; letter-spacing:-0.02em;
  line-height:1.15; margin:.15rem 0 .25rem; }
.tc-status .txt { opacity:.8; font-size:.93rem; line-height:1.5; }
.tc-green  { background:rgba(34,162,98,.10);  border-color:rgba(34,162,98,.38); }
.tc-green .ic  { background:#22a262; } .tc-green .lbl { color:#1d9a5b; }
.tc-red    { background:rgba(240,84,63,.10);  border-color:rgba(240,84,63,.38); }
.tc-red .ic    { background:#e0472f; } .tc-red .lbl { color:#e0472f; }
.tc-amber  { background:rgba(242,154,22,.12); border-color:rgba(242,154,22,.42); }
.tc-amber .ic  { background:#e08a0b; } .tc-amber .lbl { color:#d27f06; }
.tc-violet { background:rgba(124,58,237,.10); border-color:rgba(124,58,237,.36); }
.tc-violet .ic { background:#7c3aed; } .tc-violet .lbl { color:#8b5cf6; }

/* chips and small cards */
.tc-chip { display:inline-block; font-size:.78rem; font-weight:600; padding:.18rem .6rem; border-radius:999px;
  border:1px solid rgba(127,127,127,.28); margin:0 .3rem .3rem 0; white-space:nowrap; }
.tc-chip.g { color:#1d9a5b; border-color:rgba(34,162,98,.4); background:rgba(34,162,98,.08); }
.tc-chip.c { color:#0e8fb0; border-color:rgba(8,145,178,.4); background:rgba(8,145,178,.08); }
.tc-chip.r { color:#e0472f; border-color:rgba(240,84,63,.4); background:rgba(240,84,63,.08); }
.tc-chip.a { color:#d27f06; border-color:rgba(242,154,22,.45); background:rgba(242,154,22,.10); }
.tc-feature-icon { width:2.4rem; height:2.4rem; color:#fff; background:linear-gradient(135deg,#22a262,#0f5a37);
  border-radius:12px; padding:.55rem; display:inline-flex; margin-bottom:.55rem; }
.tc-big { font-family:Sora, Inter, sans-serif; font-weight:700; font-size:2.1rem; letter-spacing:-0.03em; line-height:1; }
.tc-muted { opacity:.72; }
.tc-footer { margin-top:3rem; padding-top:1.2rem; border-top:1px solid rgba(127,127,127,.22); font-size:.84rem; opacity:.8;
  line-height:1.6; }
.tc-footer a { color:inherit; }
.st-key-examples button { padding: .25rem .35rem; min-height: 2.1rem; }
.st-key-examples button p { font-size: .8rem; white-space: nowrap; }
.tc-step { font-family:Sora, Inter, sans-serif; font-weight:700; font-size:.85rem; color:#fff; background:#15834e;
  border-radius:999px; width:1.7rem; height:1.7rem; display:inline-flex; align-items:center; justify-content:center;
  margin-right:.45rem; }
@media (max-width: 640px) {
  .st-key-hero { padding: 1.6rem 1.2rem 1.4rem; border-radius: 20px; }
  [data-testid="stMainBlockContainer"] { padding-left: 1rem; padding-right: 1rem; padding-top: 4.2rem; }
}
</style>
"""


def inject() -> None:
    st.html(CSS)


def esc(text) -> str:
    return html.escape(str(text))


def section(title: str, subtitle: str | None = None, kicker: str | None = None) -> None:
    parts = []
    if kicker:
        parts.append(f'<p class="tc-kicker">{esc(kicker)}</p>')
    parts.append(f'<h2 class="tc-section-title">{esc(title)}</h2>')
    if subtitle:
        parts.append(f'<p class="tc-section-sub">{subtitle}</p>')
    st.html("".join(parts))


def chip(text: str, tone: str = "") -> str:
    return f'<span class="tc-chip {tone}">{esc(text)}</span>'


def status_card(result: dict) -> None:
    pred = result["prediction"]
    s = STATUS[pred["status"]]
    if pred["status"] in ("Healthy", "Diseased"):
        name = pred["name"]
        label = f"{s['title']} · {100 * pred['confidence']:.1f}% confidence"
    elif pred["status"] == "Uncertain":
        name = "Not sure"
        label = f"Best guess {pred['name']} · {100 * pred['confidence']:.1f}% (threshold {100 * result['confidence_threshold']:.0f}%)"
    else:
        name = "Unusual image"
        ratio = (result.get("ood") or {}).get("ratio")
        label = f"Unusualness {ratio:.2f}× the limit" if ratio else "Unusual image"
    st.html(f'<div class="tc-status tc-{s["tone"]}"><span class="ic">{icon(s["icon"])}</span><div>'
            f'<div class="lbl">{esc(label)}</div><div class="name">{esc(name)}</div>'
            f'<div class="txt">{esc(s["text"])}</div></div></div>')


def feature(name: str, title: str, body: str) -> None:
    with st.container(border=True, height="stretch"):
        st.html(f'<span class="tc-feature-icon">{icon(name)}</span>')
        st.markdown(f"**{title}**")
        st.caption(body)


def footer() -> None:
    st.html(
        '<div class="tc-footer"><b>Tomato Care</b> is a research prototype — not a substitute for an agronomist or a '
        'plant-health laboratory. Photos: Tomato-Village dataset (Gehlot, Saxena &amp; Gandhi, <i>Multimedia Systems</i> '
        f'2023, <a href="{core.DOI_URL}" target="_blank">doi:10.1007/s00530-023-01158-y</a>), used for research and '
        f'demonstration with citation. Code: <a href="{core.REPO_URL}" target="_blank">github.com/CodewithBhaumik/Tomato-Care</a> '
        '(MIT licence).</div>')
