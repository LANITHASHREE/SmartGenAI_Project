"""SmartGen AI - Resource-Aware Image Generation (Shakti Hack 26, PS1)

The problem
-----------
The district's AI image tool never asked "what is this picture for?", so every
request ran at full-poster settings and the bill tripled.

Our solution = four layers, each one removes a different kind of waste
----------------------------------------------------------------------
1. PURPOSE ROUTING  - detect coloring / flashcard / poster, use only the effort it needs
2. DRAFT FIRST      - show a tiny low-power draft; only spend full effort once the user approves it
3. LIBRARY REUSE    - if a similar picture was already made, reuse it (almost zero energy)
4. ENERGY RECEIPT   - every picture shows its energy, CO2 and what was saved vs maximum quality

Run:  pip install -r requirements.txt
      python app.py        ->  http://127.0.0.1:5000

Where to edit when a new constraint is announced
------------------------------------------------
* PRESETS / DRAFT   - settings per purpose, and the draft settings
* WH_PER_COMPUTE_UNIT, GRID_G_CO2_PER_WH, resources() - energy / CO2 estimate
* KEYWORDS          - how the purpose is detected from the prompt
* generate_image()  - swap in a real image API here
* simulate_month()  - the district simulation behind the dashboard
"""
import base64
import io
import math
import os
import random
import re
import threading
import time
import uuid
import zlib

from flask import Flask, jsonify, render_template, request, redirect, url_for, session
from PIL import Image, ImageChops, ImageColor, ImageDraw, ImageFilter, ImageFont, ImageOps

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-only-change-this-secret-before-deployment")
app.config.update(SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax")
LOGIN_USERNAME = os.environ.get("SMARTGEN_USERNAME", "admin@smartgen.ai")
LOGIN_PASSWORD = os.environ.get("SMARTGEN_PASSWORD", "SmartGen@2026")
ELECTRICITY_RUPEES_PER_KWH = float(os.environ.get("ELECTRICITY_RUPEES_PER_KWH", "8"))
PHONE_CHARGE_WH = 15.0

@app.before_request
def require_login():
    if request.endpoint in {"login", "static"}:
        return None
    if not session.get("smartgen_authenticated"):
        if request.is_json or request.path.startswith("/api/"):
            return jsonify(error="Please log in to continue."), 401
        return redirect(url_for("login", next=request.path))

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None
    if request.method == "POST":
        import hmac
        username = (request.form.get("username") or "").strip()
        password = request.form.get("password") or ""
        if hmac.compare_digest(username, LOGIN_USERNAME) and hmac.compare_digest(password, LOGIN_PASSWORD):
            session.clear()
            session["smartgen_authenticated"] = True
            session["smartgen_user"] = username
            destination = request.args.get("next", "")
            if not destination.startswith("/") or destination.startswith("//"):
                destination = url_for("home")
            return redirect(destination)
        error = "Email or password is incorrect. Please try again."
    return render_template("login.html", error=error)

@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    return redirect(url_for("login"))

# --------------------------------------------------------------------------
# 1. Purpose presets. Edit these numbers when real constraints arrive.
# --------------------------------------------------------------------------
PRESETS = {
    "coloring":  {"label": "Coloring page", "mode": "B&W outline",   "size": 512,  "steps": 15, "color": False},
    "flashcard": {"label": "Flashcard",     "mode": "Simple color",  "size": 768,  "steps": 25, "color": True},
    "poster":    {"label": "Poster",        "mode": "Full color HD", "size": 1536, "steps": 50, "color": True},
}
BASELINE = "poster"  # what the district used for EVERY request (the 3x bill)

# Low-power draft shown before the final picture is generated
DRAFT = {"label": "Draft preview", "mode": "Low-res draft", "size": 256, "steps": 8, "color": True}

# --------------------------------------------------------------------------
# 2. Resource model: compute units -> energy (Wh) -> CO2 (g). ESTIMATES.
#
# Compute grows with pixels x denoising steps. We calibrate energy so a default
# 1024px / 30-step image is about 2.9 Wh, in line with published measurements of
# diffusion-model inference (roughly 3 kWh per 1000 images). This is an ESTIMATE,
# not a meter reading. Change the constants below to match real hardware.
# --------------------------------------------------------------------------
WH_PER_COMPUTE_UNIT = 2.9 / (4 * 30)   # Wh per (512px-area x step)
GRID_G_CO2_PER_WH = 0.7                # ~0.7 kg CO2 per kWh (India grid, approx.)
SEC_PER_COMPUTE_UNIT = 4.0 / (4 * 30)  # est. seconds per unit: ~4 s for a 1024px/30-step picture on a mid-range GPU
LIBRARY_LOOKUP_WH = 0.002              # reusing a stored picture is nearly free
PHONE_CHARGE_WH = 15.0                 # one smartphone charge, rough
TREE_KG_CO2_PER_YEAR = 21.0            # one tree absorbs roughly this per year, rough


def compute_units(settings):
    pixels = (settings["size"] / 512) ** 2          # more pixels = more compute
    return pixels * settings["steps"]


def resources(settings):
    """Resource estimate for one generation: compute units, energy (Wh), CO2 (g)."""
    units = compute_units(settings)
    wh = units * WH_PER_COMPUTE_UNIT
    return {"compute": round(units, 1), "energy_wh": round(wh, 3), "time_s": round(units * SEC_PER_COMPUTE_UNIT, 2),
            "co2_g": round(wh * GRID_G_CO2_PER_WH, 3)}


def cost(settings):
    """Single number used for comparisons: estimated energy in Wh."""
    return resources(settings)["energy_wh"]


def _res_from_wh(wh, compute=0.0):
    return {"compute": round(compute, 1), "energy_wh": round(wh, 3), "co2_g": round(wh * GRID_G_CO2_PER_WH, 3)}


# --------------------------------------------------------------------------
# 3. Purpose detection from the prompt (English + Tamil + Tanglish keywords)
# --------------------------------------------------------------------------
KEYWORDS = {
    "coloring":  r"colou?ring|colou?r in|outline|worksheet|sketch|line ?art|trace|வண்ணம்|color pannu",
    "flashcard": r"flash ?card|alphabet|learning card|icon|sticker|vocabulary|number card",
    "poster":    r"poster|banner|annual report|brochure|cover|billboard|hoarding|certificate",
}


def detect_purpose(prompt):
    for purpose, pattern in KEYWORDS.items():
        if re.search(pattern, prompt, re.I):
            return purpose
    return "flashcard"  # safe middle option when we cannot tell


# --------------------------------------------------------------------------
# 4. Image generation. MOCK drawing now; swap for a real API later.
#    Everything is drawn at 3x size and scaled down (anti-aliased, smooth edges).
# --------------------------------------------------------------------------
PREVIEW_MAX = 640  # keep browser previews light, real size is shown in the labels
SS = 3             # supersampling factor for smooth edges
FLASH_FILLS = [("#ff8a80", "#ffe3e0"), ("#4fc3f7", "#dff3fd"), ("#81c784", "#e3f4e4"), ("#ffd54f", "#fff4cf")]
SUBJECTS = ("butterfly", "flower", "star", "house", "fish", "sun", "tree", "heart", "moon", "cloud")
GROUND = ("tree", "flower", "house")
_ALIASES = {"sunshine": "sun", "sunflower": "flower", "flowers": "flower", "trees": "tree", "plant": "tree",
            "stars": "star", "hearts": "heart", "home": "house", "butterflies": "butterfly"}


def _font(size):
    for name in ("segoeuib.ttf", "arialbd.ttf", "Arial Bold.ttf", "DejaVuSans-Bold.ttf",
                 "LiberationSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                 "/System/Library/Fonts/Supplemental/Arial Bold.ttf"):
        try:
            return ImageFont.truetype(name, int(size))
        except OSError:
            continue
    return ImageFont.load_default()


def _subjects(prompt):
    """All drawable things named in the prompt, in the order they appear."""
    found = []
    for m in re.finditer(r"[a-z]+", prompt.lower()):
        w = _ALIASES.get(m.group(), m.group())
        if w in SUBJECTS and w not in found:
            found.append(w)
    return found


def _subject(prompt):
    """First drawable thing in the prompt (or 'circle')."""
    s = _subjects(prompt)
    return s[0] if s else "circle"


def _title(prompt):
    t = re.sub(r"\s+", " ", prompt).strip().rstrip(".")
    return t[:1].upper() + t[1:] if t else "Untitled"


def _wrap(d, text, max_w, max_lines, start, floor, fontfn=None):
    """Largest font size (start..floor) at which text wraps into <= max_lines lines."""
    size = start
    while True:
        font = (fontfn or _font)(size)
        lines, cur = [], ""
        for word in text.split():
            trial = (cur + " " + word).strip()
            if d.textlength(trial, font=font) <= max_w or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        lines.append(cur)
        if len(lines) <= max_lines or size <= floor:
            break
        size -= max(1, int(size * 0.06))
    if len(lines) > max_lines:                      # still too long: end with an ellipsis
        lines = lines[:max_lines]
        while lines[-1] and d.textlength(lines[-1] + "...", font=font) > max_w:
            lines[-1] = lines[-1][:-1]
        lines[-1] = lines[-1].rstrip() + "..."
    return lines, font


def _gradient(img, stops):
    """Vertical multi-stop gradient. stops = [(pos0to1, (r,g,b)), ...]"""
    d = ImageDraw.Draw(img)
    w, h = img.size
    for y in range(h):
        t = y / max(1, h - 1)
        for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
            if p0 <= t <= p1:
                k = (t - p0) / (p1 - p0 or 1)
                d.line([(0, y), (w, y)], fill=tuple(int(c0[i] + (c1[i] - c0[i]) * k) for i in range(3)))
                break


def _glow(img, cx, cy, r, color, layers=24):
    """Soft glow made of translucent circles."""
    ov = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for i in range(layers, 0, -1):
        rr = r * (1 + 0.1 * i)
        od.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=color + (7,))
    return Image.alpha_composite(img.convert("RGBA"), ov)


def _sun(d, cx, cy, r, color=True, outline="black", width=3):
    if color:
        for i in range(16):
            a = i * math.pi / 8
            inner, outer = r * 1.25, r * (1.75 if i % 2 == 0 else 1.5)
            d.line([cx + math.cos(a) * inner, cy + math.sin(a) * inner,
                    cx + math.cos(a) * outer, cy + math.sin(a) * outer], fill="#ffd54f", width=max(3, int(r * 0.12)))
        d.ellipse([cx - r * 1.1, cy - r * 1.1, cx + r * 1.1, cy + r * 1.1], fill="#ffe082")
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill="#ffc107")
        d.ellipse([cx - r * 0.75, cy - r * 0.8, cx + r * 0.2, cy + r * 0.05], fill="#ffd54f")   # highlight
    else:
        for i in range(12):
            a = i * math.pi / 6
            d.line([cx + math.cos(a) * r * 1.25, cy + math.sin(a) * r * 1.25,
                    cx + math.cos(a) * r * 1.8, cy + math.sin(a) * r * 1.8], fill=outline, width=width)
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=outline, width=width)


def _tree(d, cx, base_y, h, color=True, outline="black", width=3):
    tw = h * 0.09
    if color:
        d.polygon([(cx - tw, base_y), (cx + tw, base_y), (cx + tw * 0.6, base_y - h * 0.45),
                   (cx - tw * 0.6, base_y - h * 0.45)], fill="#7a4a24")
        d.polygon([(cx + tw * 0.1, base_y), (cx + tw, base_y), (cx + tw * 0.6, base_y - h * 0.45),
                   (cx + tw * 0.1, base_y - h * 0.45)], fill="#5e3818")
    else:
        d.polygon([(cx - tw, base_y), (cx + tw, base_y), (cx + tw * 0.6, base_y - h * 0.45),
                   (cx - tw * 0.6, base_y - h * 0.45)], outline=outline, width=width)
    blobs = [(0, -0.72, 0.30), (-0.2, -0.58, 0.24), (0.2, -0.58, 0.24), (-0.1, -0.84, 0.22), (0.12, -0.80, 0.22)]
    if color:
        for dx, dy, rr in blobs:                                  # shadow, base, highlight
            x, y, r = cx + dx * h, base_y + dy * h, rr * h
            d.ellipse([x - r + h * 0.02, y - r + h * 0.03, x + r + h * 0.02, y + r + h * 0.03], fill="#237a42")
        for dx, dy, rr in blobs:
            x, y, r = cx + dx * h, base_y + dy * h, rr * h
            d.ellipse([x - r, y - r, x + r, y + r], fill="#2e9a53")
        for dx, dy, rr in blobs[:4]:
            x, y, r = cx + dx * h - h * 0.05, base_y + dy * h - h * 0.06, rr * h * 0.55
            d.ellipse([x - r, y - r, x + r, y + r], fill="#52bf74")
    else:
        for dx, dy, rr in blobs:
            x, y, r = cx + dx * h, base_y + dy * h, rr * h
            d.ellipse([x - r, y - r, x + r, y + r], outline=outline, fill="white", width=width)
        for dx, dy, rr in blobs:                                 # re-fill centre so overlaps do not show lines
            x, y, r = cx + dx * h, base_y + dy * h, rr * h * 0.85
            d.ellipse([x - r, y - r, x + r, y + r], fill="white")


def _draw_subject(d, subject, cx, cy, r, outline, fill, width):
    """Generic subject drawing (used by coloring pages, flashcards and floating poster items)."""
    f = fill
    colour = fill is not None
    if subject == "butterfly":
        for sx in (-1, 1):
            d.ellipse([cx + sx * r * 0.05 - (0 if sx > 0 else r * 0.85), cy - r * 0.8,
                       cx + sx * r * 0.05 + (r * 0.85 if sx > 0 else 0), cy + r * 0.05],
                      outline=outline, fill=f, width=width)
            d.ellipse([cx + sx * r * 0.05 - (0 if sx > 0 else r * 0.6), cy - r * 0.05,
                       cx + sx * r * 0.05 + (r * 0.6 if sx > 0 else 0), cy + r * 0.7],
                      outline=outline, fill=f, width=width)
        d.rounded_rectangle([cx - r * 0.06, cy - r * 0.6, cx + r * 0.06, cy + r * 0.6],
                            radius=r * 0.05, outline=outline, fill=outline if colour else None, width=width)
    elif subject == "flower":
        for i in range(8):
            a = i * math.pi / 4
            px, py = cx + math.cos(a) * r * 0.55, cy + math.sin(a) * r * 0.55
            d.ellipse([px - r * 0.3, py - r * 0.3, px + r * 0.3, py + r * 0.3], outline=outline, fill=f, width=width)
        d.ellipse([cx - r * 0.25, cy - r * 0.25, cx + r * 0.25, cy + r * 0.25], outline=outline,
                  fill="#f6c343" if colour else None, width=width)
    elif subject == "star":
        pts = []
        for i in range(10):
            rad = r if i % 2 == 0 else r * 0.42
            a = -math.pi / 2 + i * math.pi / 5
            pts.append((cx + math.cos(a) * rad, cy + math.sin(a) * rad))
        d.polygon(pts, outline=outline, fill=f, width=width)
    elif subject == "house":
        d.rectangle([cx - r * 0.7, cy - r * 0.1, cx + r * 0.7, cy + r * 0.7], outline=outline, fill=f, width=width)
        d.polygon([(cx - r * 0.85, cy - r * 0.1), (cx, cy - r * 0.8), (cx + r * 0.85, cy - r * 0.1)],
                  outline=outline, fill="#c0392b" if colour else None, width=width)
        d.rectangle([cx - r * 0.15, cy + r * 0.2, cx + r * 0.15, cy + r * 0.7], outline=outline,
                    fill="#8b5a2b" if colour else None, width=width)
        d.rectangle([cx + r * 0.3, cy + r * 0.05, cx + r * 0.55, cy + r * 0.3], outline=outline,
                    fill="#bfe6ff" if colour else None, width=width)
    elif subject == "fish":
        d.ellipse([cx - r * 0.8, cy - r * 0.45, cx + r * 0.5, cy + r * 0.45], outline=outline, fill=f, width=width)
        d.polygon([(cx + r * 0.4, cy), (cx + r * 0.95, cy - r * 0.4), (cx + r * 0.95, cy + r * 0.4)],
                  outline=outline, fill=f, width=width)
        d.ellipse([cx - r * 0.55, cy - r * 0.15, cx - r * 0.4, cy], outline=outline, fill=outline, width=width)
    elif subject == "sun":
        _sun(d, cx, cy, r * 0.55, color=colour, outline=outline, width=width)
    elif subject == "tree":
        _tree(d, cx, cy + r * 0.85, r * 1.7, color=colour, outline=outline, width=width)
    elif subject == "heart":
        pts = []
        for i in range(120):
            t = i * 2 * math.pi / 120
            x = 16 * math.sin(t) ** 3
            y = 13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t)
            pts.append((cx + x / 17 * r, cy - y / 17 * r + r * 0.05))
        d.polygon(pts, outline=outline, fill=f, width=width)
    elif subject == "moon":
        d.ellipse([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7], outline=outline, fill=f, width=width)
        d.ellipse([cx - r * 0.3, cy - r * 0.8, cx + r * 1.0, cy + r * 0.6], outline=outline,
                  fill="white" if not colour else "#fff4d6", width=width)
    elif subject == "cloud":
        for dx, dy, rr in ((-0.45, 0.1, 0.38), (0.0, -0.1, 0.5), (0.45, 0.1, 0.38)):
            d.ellipse([cx + dx * r - rr * r, cy + dy * r - rr * r, cx + dx * r + rr * r, cy + dy * r + rr * r],
                      outline=outline, fill=f, width=width)
        d.rectangle([cx - 0.45 * r, cy + 0.15 * r, cx + 0.45 * r, cy + 0.48 * r], fill=f)
    else:
        d.ellipse([cx - r * 0.7, cy - r * 0.7, cx + r * 0.7, cy + r * 0.7], outline=outline, fill=f, width=width)


BG_THEMES = [((22, 15, 12), (92, 62, 36)),      # warm brown (like the reference ad)
             ((7, 24, 20), (28, 92, 68)),       # deep emerald
             ((10, 15, 34), (42, 66, 128))]     # midnight blue
GOLD_STOPS = [(0, (255, 238, 178)), (0.22, (222, 168, 62)), (0.45, (255, 226, 140)),
              (0.7, (166, 111, 32)), (0.88, (240, 196, 98)), (1, (190, 135, 45))]
TEXT_GOLD = [(0, (255, 240, 190)), (0.5, (230, 182, 80)), (1, (196, 140, 48))]


def _serif(size):
    for name in ("georgiai.ttf", "timesi.ttf", "Georgia Italic.ttf", "LiberationSerif-Italic.ttf",
                 "/usr/share/fonts/truetype/liberation/LiberationSerif-Italic.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Italic.ttf",
                 "/System/Library/Fonts/Supplemental/Georgia Italic.ttf"):
        try:
            return ImageFont.truetype(name, int(size))
        except OSError:
            continue
    return _font(size)


def _sparkle(d, x, y, r, fill):
    pts = []
    for i in range(8):
        a = -math.pi / 2 + i * math.pi / 4
        rad = r if i % 2 == 0 else r * 0.16
        pts.append((x + math.cos(a) * rad, y + math.sin(a) * rad))
    d.polygon(pts, fill=fill)


def _poster(prompt, W, variant):
    """Premium dark-and-gold advert: spotlight, metallic hero, reflection, sparkles, serif headline."""
    H = int(W * 1.25)
    rng = random.Random(zlib.crc32(prompt.encode()) + variant)
    edge, centre = BG_THEMES[variant % len(BG_THEMES)]

    # 1. dark vignette background + warm spotlight + soft bokeh
    m = Image.radial_gradient("L").resize((W, H), Image.BICUBIC)
    bg = Image.composite(Image.new("RGB", (W, H), edge), Image.new("RGB", (W, H), centre), m).convert("RGBA")
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    od.ellipse([W * 0.1, H * 0.22, W * 0.9, H * 0.82], fill=(255, 196, 120, 80))
    bg = Image.alpha_composite(bg, ov.filter(ImageFilter.GaussianBlur(W * 0.1)))
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    od = ImageDraw.Draw(ov)
    for _ in range(16):
        r = W * rng.uniform(0.012, 0.045)
        x, y = rng.uniform(0, W), rng.uniform(H * 0.1, H * 0.9)
        od.ellipse([x - r, y - r, x + r, y + r], fill=(255, 205, 130, rng.randint(14, 34)))
    bg = Image.alpha_composite(bg, ov.filter(ImageFilter.GaussianBlur(W * 0.004)))

    # 2. hero artwork drawn as a silhouette with tonal detail, then turned into gold
    subs = (_subjects(prompt) or ["star"])[:3]
    layout = {1: [(0.5, 0.49, 0.28)],
              2: [(0.28, 0.48, 0.23), (0.72, 0.48, 0.23)],
              3: [(0.2, 0.52, 0.15), (0.5, 0.47, 0.21), (0.8, 0.52, 0.15)]}[len(subs)]
    src = Image.new("RGB", (W, H), "black")
    sd = ImageDraw.Draw(src)
    lw = max(3, W // 140)
    order = sorted(zip(subs, layout), key=lambda t: -t[1][2])         # biggest first, centre piece on top
    for s, (fx, fy, fr) in order[::-1] if len(subs) == 3 else order:
        _draw_subject(sd, s, W * fx, H * fy, W * fr, "#ffffff", "#9a9a9a", lw)
    lum = src.convert("L")
    mask = lum.point(lambda p: 255 if p > 6 else 0)
    bbox = mask.getbbox() or (0, 0, W, H)

    gold = Image.new("RGB", (W, H))
    _gradient(gold, GOLD_STOPS)
    shade = lum.point(lambda p: 150 + int(p * 0.41))                  # engraved detail: brighter lines
    hero = ImageChops.multiply(gold, Image.merge("RGB", (shade, shade, shade)))
    k = max(2, int(W * 0.004))
    hl = ImageChops.subtract(mask, ImageChops.offset(mask, k, k))     # top-left rim light
    sh = ImageChops.subtract(mask, ImageChops.offset(mask, -k, -k))   # bottom-right shade
    hero.paste((255, 250, 222), mask=hl)
    hero.paste((92, 56, 14), mask=sh.point(lambda p: int(p * 0.55)))
    hero = hero.convert("RGBA")
    hero.putalpha(mask)

    # warm glow around the hero, then the hero itself
    gm = mask.filter(ImageFilter.GaussianBlur(W * 0.05)).point(lambda p: int(p * 0.6))
    glow = Image.new("RGBA", (W, H), (255, 176, 80, 0))
    glow.putalpha(gm)
    bg = Image.alpha_composite(bg, glow)

    # 3. soft reflection under the hero
    gap = int(H * 0.008)
    rh = min(int(H * 0.16), H - bbox[3] - gap - 1, bbox[3] - bbox[1])
    if rh > 4:
        fl = ImageOps.flip(hero.crop((0, bbox[3] - rh, W, bbox[3])))
        fade = ImageChops.invert(Image.linear_gradient("L").resize((W, rh))).point(lambda p: int(p * 0.34))
        fl.putalpha(ImageChops.multiply(fl.getchannel("A"), fade))
        bg.alpha_composite(fl, dest=(0, bbox[3] + gap))
    bg = Image.alpha_composite(bg, hero)

    # 4. sparkles around the hero
    sp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    spd = ImageDraw.Draw(sp)
    for _ in range(11):
        x = rng.uniform(bbox[0] - W * 0.08, bbox[2] + W * 0.08)
        y = rng.uniform(bbox[1] - H * 0.06, bbox[3] + H * 0.01)
        _sparkle(spd, x, y, W * rng.uniform(0.008, 0.03), (255, 246, 220, 240))
    bg = Image.alpha_composite(bg, sp.filter(ImageFilter.GaussianBlur(W * 0.006)))
    bg = Image.alpha_composite(bg, sp)
    img = bg.convert("RGB")

    # 5. typography: gold italic serif headline, thin rule, short body, small brand mark
    d = ImageDraw.Draw(img)
    lines, font = _wrap(d, _title(prompt), W * 0.84, 3, W * 0.082, W * 0.045, _serif)
    lh = font.size * 1.2
    x0, y0 = W * 0.08, H * 0.065
    block = int(lh * len(lines) + font.size * 0.3)
    tm = Image.new("L", (W, block), 0)
    td = ImageDraw.Draw(tm)
    for i, line in enumerate(lines):
        td.text((x0, i * lh), line, fill=255, font=font, anchor="la")
    tg = Image.new("RGB", (W, block))
    _gradient(tg, TEXT_GOLD)
    img.paste(tg, (0, int(y0)), tm)
    ry = y0 + block + H * 0.008
    d.rectangle([x0, ry, x0 + W * 0.14, ry + max(2, W // 320)], fill=(214, 170, 80))

    body = ("Every detail considered,", "every watt accounted for,", "every picture made to matter.")
    bf = _serif(W * 0.03)
    for i, line in enumerate(body):
        d.text((W * 0.92, H * 0.815 + i * bf.size * 1.3), line, fill=(214, 198, 170), font=bf, anchor="ra")

    lr, ly = W * 0.032, H * 0.94
    d.ellipse([x0, ly - lr, x0 + lr * 2, ly + lr], outline=(222, 176, 84), width=max(2, W // 400))
    d.text((x0 + lr, ly), "S", fill=(240, 205, 120), font=_serif(lr * 1.5), anchor="mm")
    d.text((x0 + lr * 2.7, ly), "SMARTGEN AI", fill=(214, 190, 150), font=_font(W * 0.022), anchor="lm")
    return img


def _flashcard(prompt, W, variant):
    subs = _subjects(prompt)
    subject = subs[0] if subs else "circle"
    accent, soft = FLASH_FILLS[variant % len(FLASH_FILLS)]
    img = Image.new("RGB", (W, W), soft)
    d = ImageDraw.Draw(img)
    m = W * 0.06
    d.rounded_rectangle([m + W * 0.012, m + W * 0.016, W - m + W * 0.012, W - m + W * 0.016], radius=W * 0.06,
                        fill=tuple(int(c * 0.82) for c in ImageColor.getrgb(soft)))   # soft shadow
    d.rounded_rectangle([m, m, W - m, W - m], radius=W * 0.06, fill="white", outline=accent, width=max(3, W // 70))
    _draw_subject(d, subject, W / 2, W * 0.42, W * 0.27, "#2d3436", accent, max(3, W // 110))
    label = subject if subject != "circle" else prompt[:18]
    lines, font = _wrap(d, label.upper(), W * 0.7, 1, W * 0.1, W * 0.05)
    d.text((W / 2, W * 0.80), lines[0], fill="#2d3436", font=font, anchor="mm")
    d.rounded_rectangle([W / 2 - W * 0.08, W * 0.875, W / 2 + W * 0.08, W * 0.885], radius=W * 0.005, fill=accent)
    return img


def _coloring(prompt, W, variant):
    subs = _subjects(prompt)
    subject = subs[0] if subs else "circle"
    img = Image.new("RGB", (W, W), "white")
    d = ImageDraw.Draw(img)
    lw = max(4, W // 110)
    d.rounded_rectangle([W * 0.04, W * 0.04, W * 0.96, W * 0.96], radius=W * 0.04, outline="black", width=lw)
    _draw_subject(d, subject, W / 2, W * 0.46, W * 0.32 * (1 + 0.04 * (variant % 3)), "black", None, lw)
    lines, font = _wrap(d, "Colour me: " + subject.title() if subject != "circle" else _title(prompt),
                        W * 0.8, 1, W * 0.05, W * 0.03)
    d.text((W / 2, W * 0.89), lines[0], fill="black", font=font, anchor="mm")
    return img


def generate_image(prompt, settings, style=None, variant=0, tag="MOCK", use_api=True):
    """MOCK generator. Draws a clean, professional-looking picture whose style follows the settings.

    style   : "coloring" | "flashcard" | "poster" (decides how it looks)
    variant : 0, 1, 2... gives a slightly different picture (used for new drafts)
    tag     : "DRAFT" adds a small DRAFT badge; anything else adds nothing

    To use a real image API, replace the body of this function: send `prompt`
    plus `settings['size']` and `settings['steps']` to the API, then return the
    result as a data URL:  "data:image/png;base64,...."
    """
    if style is None:
        style = "coloring" if not settings["color"] else ("flashcard" if settings["size"] < 1000 else "poster")
    if use_api and tag != "DRAFT" and ai_enabled():
        try:
            real = _real_image(prompt, settings, style)
            if style == "poster":
                real = _poster_overlay(real, prompt)
            elif style == "coloring":
                # Enforce clean black ink on a pure white page for coloring sheets.
                gray = ImageOps.grayscale(real)
                real = gray.point(lambda px: 0 if px < 175 else 255, mode="1").convert("RGB")
            buf = io.BytesIO()
            real.save(buf, format="PNG", optimize=True)
            return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        except Exception as exc:                       # no network / quota / model not served -> mock
            print(f"[SmartGen] real image failed ({type(exc).__name__}: {str(exc)[:120]}), using mock drawing")
    side = min(settings["size"], PREVIEW_MAX)
    W = side * (2 if style == "poster" else SS)
    img = {"coloring": _coloring, "flashcard": _flashcard}.get(style, _poster)(prompt, W, variant)
    img = img.resize((side, round(side * img.height / img.width)), Image.LANCZOS)   # poster is 4:5 portrait

    if tag == "DRAFT":
        d = ImageDraw.Draw(img)
        f = _font(max(9, side // 20))
        tw = d.textlength("DRAFT", font=f)
        pad = max(4, side // 40)
        d.rounded_rectangle([pad, pad, pad * 3 + tw, pad * 3 + f.size * 0.7], radius=pad, fill="#2d3436")
        d.text((pad * 2, pad * 2 + f.size * 0.35), "DRAFT", fill="white", font=f, anchor="lm")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


# --------------------------------------------------------------------------
# 4b. REAL AI image generation (photo-realistic). Used for FINAL pictures when a token is set.
#     export HF_TOKEN=hf_xxx        (free token: huggingface.co/settings/tokens)
#     export AI_MODEL=black-forest-labs/FLUX.1-schnell   (optional, any text-to-image model your
#                                                          Hugging Face Inference Provider serves)
#     No token -> the built-in mock drawing is used, so the demo never breaks.
#     Drafts and the side-by-side "full quality" preview always use the cheap mock (that IS the saving).
# --------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_TOKEN", "")
AI_MODEL = os.environ.get("AI_MODEL", "black-forest-labs/FLUX.1-schnell")
REAL_STYLE = {
    "poster": ("premium advertising poster photograph, dramatic cinematic studio lighting, dark elegant "
               "background with soft golden bokeh, glossy reflections, ultra detailed, sharp focus, "
               "professional commercial photography, centered hero subject, empty space at the top, "
               "no text, no letters, no watermark"),
    "flashcard": ("cute children's educational flashcard illustration, single subject centered, clean plain "
                  "pastel background, bold vibrant colours, smooth vector style, no text, no letters"),
    "coloring": ("black and white coloring book page, clean thick outlines, simple shapes, pure white "
                 "background, no shading, no grey, no text"),
}
REAL_NEGATIVE = "text, letters, watermark, blurry, low quality, distorted, deformed, extra limbs, cropped"


def ai_enabled():
    return bool(HF_TOKEN)


def _round8(n):
    return max(256, int(n) // 8 * 8)


def _real_image(prompt, settings, style):
    """Ask a real text-to-image model. Size/steps come from the purpose settings (capped for the API)."""
    from huggingface_hub import InferenceClient          # pip install huggingface_hub
    long_side = min(settings["size"], 1024)
    w, h = (_round8(long_side * 0.8), _round8(long_side)) if style == "poster" else (_round8(long_side),) * 2
    kwargs = {"model": AI_MODEL, "width": w, "height": h}
    if "xl" in AI_MODEL.lower():                          # SDXL-type models accept these; FLUX-schnell does not
        kwargs.update(negative_prompt=REAL_NEGATIVE, num_inference_steps=min(settings["steps"], 40))
    client = InferenceClient(token=HF_TOKEN, timeout=120)
    return client.text_to_image(f"{prompt}, {REAL_STYLE[style]}", **kwargs).convert("RGB")


def _poster_overlay(img, prompt):
    """Crisp typography on top of the AI picture (AI models spell badly, so text is added by us)."""
    W, H = img.size
    img = img.convert("RGBA")
    fade = Image.linear_gradient("L").transpose(Image.FLIP_TOP_BOTTOM).resize((W, int(H * 0.42)))
    shade = Image.new("RGBA", (W, H), (8, 6, 4, 0))
    shade.paste(Image.new("RGBA", (W, fade.height), (8, 6, 4, 255)), (0, 0), fade.point(lambda p: int(p * 0.82)))
    img = Image.alpha_composite(img, shade)
    bottom = Image.linear_gradient("L").resize((W, int(H * 0.16)))
    sh2 = Image.new("RGBA", (W, H), (8, 6, 4, 0))
    sh2.paste(Image.new("RGBA", (W, bottom.height), (8, 6, 4, 255)), (0, H - bottom.height),
              bottom.point(lambda p: int(p * 0.75)))
    img = Image.alpha_composite(img, sh2).convert("RGB")

    d = ImageDraw.Draw(img)
    lines, font = _wrap(d, _title(prompt), W * 0.84, 3, W * 0.08, W * 0.045, _serif)
    lh = font.size * 1.2
    x0, y0 = W * 0.08, H * 0.055
    block = int(lh * len(lines) + font.size * 0.3)
    tm = Image.new("L", (W, block), 0)
    td = ImageDraw.Draw(tm)
    for i, line in enumerate(lines):
        td.text((x0, i * lh), line, fill=255, font=font, anchor="la")
    tg = Image.new("RGB", (W, block))
    _gradient(tg, TEXT_GOLD)
    img.paste(tg, (0, int(y0)), tm)
    ry = y0 + block + H * 0.008
    d.rectangle([x0, ry, x0 + W * 0.14, ry + max(2, W // 320)], fill=(214, 170, 80))
    lr, ly = W * 0.032, H * 0.95
    d.ellipse([x0, ly - lr, x0 + lr * 2, ly + lr], outline=(222, 176, 84), width=max(2, W // 400))
    d.text((x0 + lr, ly), "S", fill=(240, 205, 120), font=_serif(lr * 1.5), anchor="mm")
    d.text((x0 + lr * 2.7, ly), "SMARTGEN AI", fill=(214, 190, 150), font=_font(W * 0.022), anchor="lm")
    return img


# --------------------------------------------------------------------------
# 5. Picture library (reuse) - the same idea is never generated twice
# --------------------------------------------------------------------------
_STOP = {"a", "an", "the", "of", "with", "and", "for", "to", "please", "make", "create", "generate",
         "me", "free", "kids", "kid", "children", "child", "page", "sheet", "worksheet", "picture",
         "image", "photo", "drawing", "my", "our", "in", "on"}
_LIBRARY = {}   # key -> {"tokens", "purpose", "prompt", "image", "hits"}
_LOCK = threading.Lock()


def _tokens(prompt):
    return frozenset(t for t in re.findall(r"[a-z0-9]+", prompt.lower()) if t not in _STOP)


def library_lookup(prompt, purpose):
    toks = _tokens(prompt)
    if not toks:
        return None
    with _LOCK:
        for entry in _LIBRARY.values():
            if entry["purpose"] != purpose:
                continue
            union = toks | entry["tokens"]
            if len(toks & entry["tokens"]) / len(union) >= 0.8:
                entry["hits"] += 1
                return dict(entry)
    return None


def library_store(prompt, purpose, image):
    toks = _tokens(prompt)
    if not toks:
        return
    with _LOCK:
        _LIBRARY.setdefault((purpose, toks), {"tokens": toks, "purpose": purpose, "prompt": prompt,
                                              "image": image, "hits": 0})


# --------------------------------------------------------------------------
# 6. Usage log + draft flows: real savings from pictures made in this session
# --------------------------------------------------------------------------
_LOG = []
_FLOWS = {}     # flow_id -> {"prompt", "purpose", "attempts", "spent_wh"}


def _record(purpose, source, spent_wh, baseline_wh, drafts=0):
    with _LOCK:
        _LOG.append({"t": time.strftime("%H:%M:%S"), "purpose": purpose, "source": source,
                     "smart": round(spent_wh, 3), "baseline": round(baseline_wh, 3), "drafts": drafts})


def _explain(purpose, settings, auto):
    if auto:
        reason = (f"Detected '{settings['label']}' from your prompt, so we used "
                  f"{settings['mode']} at {settings['size']}px / {settings['steps']} steps")
        return reason + ("." if purpose == BASELINE else " instead of full poster settings.")
    return f"You chose '{settings['label']}', so we used {settings['mode']} settings."


def _fmt_units(settings):
    return f"({settings['size']}/512)\u00b2 \u00d7 {settings['steps']} steps = {compute_units(settings):g} units"


def _how(purpose, source, attempts, spent_wh, baseline_wh):
    """Plain-language, step-by-step explanation of THIS picture's estimate (shown to the user)."""
    S, B = PRESETS[purpose], PRESETS[BASELINE]
    lines = [f"Compute units = (size \u00f7 512)\u00b2 \u00d7 steps. More pixels and more steps = more work for the model.",
             f"Energy = units \u00d7 {WH_PER_COMPUTE_UNIT:.4f} Wh. Calibrated so a 1024px / 30-step picture is about 2.9 Wh "
             f"(published diffusion-model measurements are roughly 3 kWh per 1000 images).",
             f"Time (estimate) = units \u00d7 {SEC_PER_COMPUTE_UNIT:.3f} s (about 4 s for 1024px / 30 steps on a mid-range GPU).",
             f"CO\u2082 = energy \u00d7 {GRID_G_CO2_PER_WH} g per Wh (India grid, about 0.7 kg per kWh)."]
    if source == "library":
        lines.append(f"This picture: a similar one already existed, so only a library lookup was needed, fixed at {LIBRARY_LOOKUP_WH} Wh. "
                     f"Maximum quality would have generated a new {B['label'].lower()}: {_fmt_units(B)} = {baseline_wh:.2f} Wh.")
    elif source == "draft+final":
        lines.append(f"This picture: {attempts} draft(s) \u00d7 {cost(DRAFT):.3f} Wh ({_fmt_units(DRAFT)}) + 1 final {S['label'].lower()} "
                     f"({_fmt_units(S)} = {cost(S):.2f} Wh) = {spent_wh:.2f} Wh.")
        lines.append(f"Maximum quality would make a full {B['label'].lower()} for each of the {attempts} tries: "
                     f"{attempts} \u00d7 {cost(B):.2f} Wh ({_fmt_units(B)}) = {baseline_wh:.2f} Wh.")
    else:
        lines.append(f"This picture: {_fmt_units(S)} \u2192 {spent_wh:.2f} Wh. "
                     f"Maximum quality ({B['label']}): {_fmt_units(B)} \u2192 {baseline_wh:.2f} Wh.")
    pct = (1 - spent_wh / baseline_wh) * 100 if baseline_wh else 0
    lines.append(f"Saving = maximum \u2212 smart = {baseline_wh - spent_wh:.2f} Wh ({pct:.1f}%).")
    lines.append("These are estimates, not meter readings: we cannot meter the GPU. Measured time is the real stopwatch time on this server.")
    if ai_enabled():
        lines.append(f"Real AI model ({AI_MODEL}) is connected: the provider caps size at 1024px, but we still estimate from the purpose "
                     f"settings, so the figure is conservative and the same formula is used for smart and maximum.")
    return lines


def _final_payload(prompt, purpose, auto, image, spent_wh, baseline_wh, source, attempts=1, drafts=0, note="", measured_s=0.0):
    settings = PRESETS[purpose]
    spent, base = _res_from_wh(spent_wh), _res_from_wh(baseline_wh)
    spent["compute"] = round(spent_wh / WH_PER_COMPUTE_UNIT, 1)
    base["compute"] = round(baseline_wh / WH_PER_COMPUTE_UNIT, 1)
    spent["time_s"] = round(spent["compute"] * SEC_PER_COMPUTE_UNIT, 3)
    base["time_s"] = round(base["compute"] * SEC_PER_COMPUTE_UNIT, 3)
    saved = {"energy_wh": round(baseline_wh - spent_wh, 3),
             "co2_g": round((baseline_wh - spent_wh) * GRID_G_CO2_PER_WH, 3),
             "time_s": round(base["time_s"] - spent["time_s"], 3)}
    return dict(
        image=image,
        baseline_image=generate_image(prompt, PRESETS[BASELINE], style=BASELINE, use_api=False),  # side-by-side preview
        purpose=purpose, auto_detected=auto, settings=settings, reason=_explain(purpose, settings, auto),
        source=source, attempts=attempts, drafts=drafts, note=note,
        smart=spent, baseline=base, saved=saved, how=_how(purpose, source, attempts, spent_wh, baseline_wh),
        measured_s=round(measured_s, 2),
        smart_cost=spent["energy_wh"], baseline_cost=base["energy_wh"],
        saved_pct=round((1 - spent_wh / baseline_wh) * 100, 1) if baseline_wh else 0,
    )


def _read_prompt(data):
    prompt = (data.get("prompt") or "").strip()
    if not prompt:
        return None, None, None, (jsonify(error="Prompt is empty"), 400)
    if len(prompt) > 300:
        return None, None, None, (jsonify(error="Prompt is too long (max 300 characters)"), 400)
    purpose = data.get("purpose") or "auto"
    if purpose != "auto" and purpose not in PRESETS:
        return None, None, None, (jsonify(error=f"Unknown purpose '{purpose}'"), 400)
    auto = purpose == "auto"
    if auto:
        purpose = detect_purpose(prompt)
    return prompt, purpose, auto, None


# --------------------------------------------------------------------------
# 7. District simulation behind the dashboard
# --------------------------------------------------------------------------
DEFAULT_MIX = {"coloring": 0.667, "flashcard": 0.278, "poster": 0.055}  # from the audit: mostly coloring
TIER_CODE = {"coloring": 0, "flashcard": 1, "poster": 2}


def simulate_month(schools=300, per_school=30, avg_attempts=2.0, reuse_pct=30, seed=7, days=30):
    """Simulate one month of requests from every school under four scenarios.

    A  always maximum quality (what the district did)
    B  + purpose routing
    C  + draft first (every attempt is a tiny draft, only the approved one is made in full)
    D  + library reuse (similar pictures are reused instead of regenerated)

    ASSUMPTIONS (shown on the dashboard, adjustable): requests per school,
    attempts before a user is happy, and how often a similar picture already exists.
    """
    rng = random.Random(seed)
    schools = max(1, min(int(schools), 1000))
    per_school = max(1, min(int(per_school), 200))
    avg_attempts = max(1.0, min(float(avg_attempts), 6.0))
    reuse_pct = max(0.0, min(float(reuse_pct), 90.0))
    p_accept = 1.0 / avg_attempts

    P = cost(PRESETS[BASELINE])
    T = {k: cost(v) for k, v in PRESETS.items()}
    D = cost(DRAFT)

    weights = [0.4 + rng.random() * 1.2 for _ in range(schools)]       # some schools use it more
    cum = []
    run = 0.0
    for w in weights:
        run += w
        cum.append(run)
    day_w = [0.25 if (d % 7) in (5, 6) else 1.0 for d in range(days)]  # quiet weekends
    total_requests = schools * per_school
    counts = [round(total_requests * w / sum(day_w)) for w in day_w]

    cumulative = {"a": [], "b": [], "c": [], "d": []}
    tot = {"a": 0.0, "b": 0.0, "c": 0.0, "d": 0.0}
    events, tier_counts, hits = [], {k: 0 for k in PRESETS}, 0
    school_a, school_d, school_n = [0.0] * schools, [0.0] * schools, [0] * schools

    for d in range(days):
        ramp = min(1.0, (d + 1) / 10) * reuse_pct / 100            # library fills up over the first days
        day_events = []
        for _ in range(counts[d]):
            s = rng.choices(range(schools), cum_weights=cum)[0]
            purpose = rng.choices(list(DEFAULT_MIX), weights=list(DEFAULT_MIX.values()))[0]
            attempts = 1
            while attempts < 8 and rng.random() > p_accept:
                attempts += 1
            a = attempts * P
            b = attempts * T[purpose]
            c = attempts * D + T[purpose]
            hit = rng.random() < ramp
            dd = LIBRARY_LOOKUP_WH if hit else c
            tot["a"] += a; tot["b"] += b; tot["c"] += c; tot["d"] += dd
            school_a[s] += a; school_d[s] += dd; school_n[s] += 1
            tier_counts[purpose] += 1
            hits += 1 if hit else 0
            day_events.append(s * 8 + TIER_CODE[purpose] * 2 + (1 if hit else 0))
        events.append(day_events)
        for k in cumulative:
            cumulative[k].append(round(tot[k] / 1000, 3))           # kWh

    saved_wh = tot["a"] - tot["d"]
    top = sorted(range(schools), key=lambda i: school_a[i] - school_d[i], reverse=True)[:5]
    return {
        "days": days, "schools": schools, "requests": sum(counts),
        "cumulative_kwh": cumulative,
        "totals_kwh": {k: round(v / 1000, 2) for k, v in tot.items()},
        "breakdown_kwh": {"routing": round((tot["a"] - tot["b"]) / 1000, 2),
                          "draft": round((tot["b"] - tot["c"]) / 1000, 2),
                          "reuse": round((tot["c"] - tot["d"]) / 1000, 2)},
        "saved_kwh": round(saved_wh / 1000, 2),
        "saved_pct": round(saved_wh / tot["a"] * 100, 1) if tot["a"] else 0,
        "co2_kg": round(saved_wh * GRID_G_CO2_PER_WH / 1000, 1),
        "phone_charges": int(saved_wh / PHONE_CHARGE_WH),
        "tree_years": round(saved_wh * GRID_G_CO2_PER_WH / 1000 / TREE_KG_CO2_PER_YEAR, 1),
        "tier_counts": tier_counts, "reused": hits,
        "factors": {"co2_kg_per_kwh": GRID_G_CO2_PER_WH, "phone_wh": PHONE_CHARGE_WH,
                    "tree_kg_per_year": TREE_KG_CO2_PER_YEAR},
        "events": events,
        "top_schools": [{"name": f"School {i + 1:03d}", "requests": school_n[i],
                         "saved_kwh": round((school_a[i] - school_d[i]) / 1000, 2),
                         "saved_pct": round((1 - school_d[i] / school_a[i]) * 100, 1) if school_a[i] else 0}
                        for i in top],
        "assumptions": {"per_school": per_school, "avg_attempts": avg_attempts, "reuse_pct": reuse_pct,
                        "poster_wh": round(P, 2), "draft_wh": round(D, 3),
                        "tier_wh": {k: round(v, 2) for k, v in T.items()}},
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def home():
    return render_template("index.html", presets=PRESETS, draft=DRAFT, draft_wh=cost(DRAFT),
                           method=dict(wh=WH_PER_COMPUTE_UNIT, sec=SEC_PER_COMPUTE_UNIT, co2=GRID_G_CO2_PER_WH,
                                       lookup=LIBRARY_LOOKUP_WH, max_wh=cost(PRESETS[BASELINE]),
                                       max_units=compute_units(PRESETS[BASELINE])))


@app.route("/estimate", methods=["POST"])
def estimate():
    """Return resource estimates only; this endpoint never generates an image."""
    data = request.get_json(force=True, silent=True) or {}
    prompt, purpose, auto, err = _read_prompt(data)
    if err:
        return err
    settings = PRESETS[purpose]
    smart, maximum = resources(settings), resources(PRESETS[BASELINE])
    smart["cost_inr"] = round(smart["energy_wh"] / 1000 * ELECTRICITY_RUPEES_PER_KWH, 6)
    maximum["cost_inr"] = round(maximum["energy_wh"] / 1000 * ELECTRICITY_RUPEES_PER_KWH, 6)
    saved_wh = max(0.0, maximum["energy_wh"] - smart["energy_wh"])
    return jsonify(prompt=prompt, purpose=purpose, auto_detected=auto, settings=settings,
        smart=smart, maximum=maximum, saved_energy_wh=round(saved_wh, 3),
        saved_pct=round((1-smart["energy_wh"]/maximum["energy_wh"])*100, 1) if maximum["energy_wh"] else 0,
        saved_cost_inr=round(saved_wh/1000*ELECTRICITY_RUPEES_PER_KWH, 6),
        electricity_rate_inr_kwh=ELECTRICITY_RUPEES_PER_KWH,
        equivalent_phone_charge_pct=round(smart["energy_wh"]/PHONE_CHARGE_WH*100, 2),
        reason=_explain(purpose, settings, auto), estimate_only=True)


@app.route("/draft", methods=["POST"])
def draft():
    """Step 1: tiny low-power draft (or an instant library hit)."""
    data = request.get_json(force=True, silent=True) or {}
    prompt, purpose, auto, err = _read_prompt(data)
    if err:
        return err
    flow_id = data.get("flow_id") or uuid.uuid4().hex
    P = cost(PRESETS[BASELINE])
    with _LOCK:
        flow = _FLOWS.setdefault(flow_id, {"prompt": prompt, "purpose": purpose, "attempts": 0, "spent_wh": 0.0})
        attempts_before = flow["attempts"]

    if attempts_before == 0:
        hit = library_lookup(prompt, purpose)
        if hit:
            _record(purpose, "library", LIBRARY_LOOKUP_WH, P)
            with _LOCK:
                _FLOWS.pop(flow_id, None)
            return jsonify(done=True, flow_id=flow_id, **_final_payload(
                prompt, purpose, auto, hit["image"], LIBRARY_LOOKUP_WH, P, "library",
                note=f"A similar picture ('{hit['prompt']}') already exists in the library, so nothing was generated."))

    t0 = time.perf_counter()
    draft_img = generate_image(prompt, DRAFT, style=purpose, variant=attempts_before, tag="DRAFT")
    took = time.perf_counter() - t0
    with _LOCK:
        flow["attempts"] += 1
        flow["spent_wh"] += cost(DRAFT)
        flow["measured_s"] = flow.get("measured_s", 0.0) + took
        flow["prompt"], flow["purpose"] = prompt, purpose
        attempts, spent = flow["attempts"], flow["spent_wh"]
    return jsonify(
        draft_wh=cost(DRAFT), max_wh=P, draft_saved_pct=round((1 - cost(DRAFT) / P) * 100, 1),
        draft_time_s=round(resources(DRAFT)["time_s"], 3), max_time_s=round(resources(PRESETS[BASELINE])["time_s"], 2),
        measured_s=round(took, 3), draft_how=f"Draft = {_fmt_units(DRAFT)} \u00d7 {WH_PER_COMPUTE_UNIT:.4f} Wh = {cost(DRAFT):.3f} Wh. "
        f"One maximum-quality picture ({_fmt_units(PRESETS[BASELINE])}) = {P:.2f} Wh.",
        done=False, flow_id=flow_id, purpose=purpose, auto_detected=auto,
        settings=PRESETS[purpose], draft_settings=DRAFT, reason=_explain(purpose, PRESETS[purpose], auto),
        image=draft_img,
        attempts=attempts, spent_wh=round(spent, 3), would_have_spent_wh=round(attempts * P, 3),
        final_wh=cost(PRESETS[purpose]),
    )


@app.route("/finalize", methods=["POST"])
def finalize():
    """Step 2: the user approved the draft, so spend the effort the purpose needs."""
    data = request.get_json(force=True, silent=True) or {}
    flow_id = data.get("flow_id")
    with _LOCK:
        flow = _FLOWS.pop(flow_id, None)
    if not flow:
        return jsonify(error="Draft not found. Please generate a draft first."), 400
    prompt, purpose, attempts = flow["prompt"], flow["purpose"], flow["attempts"]
    auto = bool(data.get("auto_detected", True))
    settings = PRESETS[purpose]
    t0 = time.perf_counter()
    image = generate_image(prompt, settings, style=purpose, variant=attempts - 1)
    measured = flow.get("measured_s", 0.0) + (time.perf_counter() - t0)
    library_store(prompt, purpose, image)
    spent = flow["spent_wh"] + cost(settings)
    baseline = attempts * cost(PRESETS[BASELINE])   # a baseline tool makes a full poster for every attempt
    _record(purpose, "draft+final", spent, baseline, drafts=attempts)
    return jsonify(done=True, flow_id=flow_id, **_final_payload(
        prompt, purpose, auto, image, spent, baseline, "draft+final", attempts=attempts, drafts=attempts, measured_s=measured,
        note=f"{attempts} low-power draft{'s' if attempts != 1 else ''} + 1 final picture. "
             f"Without drafts, every try would have been a full poster."))


@app.route("/generate", methods=["POST"])
def generate():
    """Skip the draft step: generate the final picture directly at the purpose's settings."""
    data = request.get_json(force=True, silent=True) or {}
    prompt, purpose, auto, err = _read_prompt(data)
    if err:
        return err
    settings = PRESETS[purpose]
    P = cost(PRESETS[BASELINE])
    hit = library_lookup(prompt, purpose)
    if hit:
        _record(purpose, "library", LIBRARY_LOOKUP_WH, P)
        return jsonify(**_final_payload(prompt, purpose, auto, hit["image"], LIBRARY_LOOKUP_WH, P, "library",
                                        note=f"A similar picture ('{hit['prompt']}') already exists in the library."))
    t0 = time.perf_counter()
    image = generate_image(prompt, settings, style=purpose)
    took = time.perf_counter() - t0
    library_store(prompt, purpose, image)
    _record(purpose, "final", cost(settings), P)
    return jsonify(**_final_payload(prompt, purpose, auto, image, cost(settings), P, "final", measured_s=took))


@app.route("/simulate", methods=["POST"])
def simulate():
    """Quick calculator: N requests split by purpose -> total energy, smart vs baseline."""
    data = request.get_json(force=True, silent=True) or {}
    counts = {}
    for k in PRESETS:
        try:
            counts[k] = max(0, int(data.get(k, 0)))
        except (TypeError, ValueError):
            counts[k] = 0
    smart = sum(cost(PRESETS[k]) * n for k, n in counts.items())
    base = cost(PRESETS[BASELINE]) * sum(counts.values())
    saved_wh = base - smart
    return jsonify(smart=round(smart, 2), baseline=round(base, 2),          # Wh
                   smart_kwh=round(smart / 1000, 2), baseline_kwh=round(base / 1000, 2),
                   saved_kwh=round(saved_wh / 1000, 2),
                   saved_co2_kg=round(saved_wh * GRID_G_CO2_PER_WH / 1000, 2),
                   saved_pct=round((1 - smart / base) * 100, 1) if base else 0)


@app.route("/simulate_month", methods=["POST"])
def simulate_month_route():
    data = request.get_json(force=True, silent=True) or {}
    try:
        result = simulate_month(schools=data.get("schools", 300), per_school=data.get("per_school", 30),
                                avg_attempts=data.get("avg_attempts", 2.0), reuse_pct=data.get("reuse_pct", 30))
    except (TypeError, ValueError):
        return jsonify(error="Invalid simulation settings"), 400
    return jsonify(result)


@app.route("/stats")
def stats():
    """Real usage so far: what we spent vs what 'always maximum quality' would have cost."""
    with _LOCK:
        log = list(_LOG)
        library_size = len(_LIBRARY)
    smart = round(sum(r["smart"] for r in log), 2)
    base = round(sum(r["baseline"] for r in log), 2)
    by_purpose = {k: sum(1 for r in log if r["purpose"] == k) for k in PRESETS}
    return jsonify(requests=len(log), smart=smart, baseline=base,
                   saved_wh=round(base - smart, 2),
                   saved_co2_g=round((base - smart) * GRID_G_CO2_PER_WH, 2),
                   saved_pct=round((1 - smart / base) * 100, 1) if base else 0,
                   reused=sum(1 for r in log if r["source"] == "library"),
                   drafts=sum(r["drafts"] for r in log), library_size=library_size,
                   by_purpose=by_purpose, recent=log[-5:][::-1])


@app.route("/stats/reset", methods=["POST"])
def stats_reset():
    with _LOCK:
        _LOG.clear()
        _LIBRARY.clear()
        _FLOWS.clear()
    return jsonify(ok=True)


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "1") == "1")
