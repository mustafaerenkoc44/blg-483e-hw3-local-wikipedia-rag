"""Generate a YouTube thumbnail (1280x720) for the demo video.

Run::

    python scripts/make_thumbnail.py

Output: ``thumbnail.png`` in the repository root. Upload this directly when
creating the unlisted YouTube video.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

WIDTH, HEIGHT = 1280, 720
OUTPUT = Path(__file__).resolve().parent.parent / "thumbnail.png"

BG_TOP = (8, 10, 18)
BG_MID = (16, 20, 32)
BG_BOTTOM = (24, 28, 44)
ACCENT = (255, 75, 75)
ACCENT_DIM = (200, 60, 60)
TEXT_PRIMARY = (250, 250, 250)
TEXT_DIM = (170, 178, 200)
CARD_BG = (28, 32, 48)
CARD_BORDER = (70, 78, 110)
GRID_DOT = (40, 46, 70)


def vertical_gradient(size, stops):
    """Build a vertical gradient with N stops at evenly spaced positions."""
    img = Image.new("RGB", size, stops[0])
    px = img.load()
    w, h = size
    n = len(stops) - 1
    for y in range(h):
        t = y / max(1, h - 1)
        seg = min(int(t * n), n - 1)
        local_t = (t * n) - seg
        a, b = stops[seg], stops[seg + 1]
        r = int(a[0] + (b[0] - a[0]) * local_t)
        g = int(a[1] + (b[1] - a[1]) * local_t)
        bl = int(a[2] + (b[2] - a[2]) * local_t)
        for x in range(w):
            px[x, y] = (r, g, bl)
    return img


def add_dot_grid(img, spacing=40, color=GRID_DOT):
    draw = ImageDraw.Draw(img)
    for y in range(0, img.height, spacing):
        for x in range(0, img.width, spacing):
            draw.ellipse([x - 1, y - 1, x + 1, y + 1], fill=color)
    return img


def add_glow(img, center, radius, color, opacity=80):
    """Add a soft radial glow centered at ``center`` with given ``radius``."""
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    o = ImageDraw.Draw(overlay)
    cx, cy = center
    o.ellipse(
        [cx - radius, cy - radius, cx + radius, cy + radius],
        fill=(*color, opacity),
    )
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=radius // 2))
    return Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB")


def load_font(size, bold=False):
    candidates = [
        f"C:/Windows/Fonts/{'segoeuib' if bold else 'segoeui'}.ttf",
        f"C:/Windows/Fonts/{'arialbd' if bold else 'arial'}.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def text_size(draw, text, font):
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def main() -> int:
    img = vertical_gradient((WIDTH, HEIGHT), [BG_TOP, BG_MID, BG_BOTTOM])
    img = add_dot_grid(img, spacing=42)
    img = add_glow(img, (180, 320), 360, ACCENT, opacity=70)
    img = add_glow(img, (1100, 540), 320, (60, 120, 220), opacity=50)
    draw = ImageDraw.Draw(img)

    # Top tag
    tag_font = load_font(24, bold=True)
    tag = "BLG 483E  ·  HOMEWORK 3"
    draw.text((60, 48), tag, font=tag_font, fill=ACCENT)

    # Big title block
    title_font = load_font(118, bold=True)
    draw.text((60, 92), "LOCAL", font=title_font, fill=TEXT_PRIMARY)
    draw.text((60, 222), "WIKIPEDIA", font=title_font, fill=TEXT_PRIMARY)
    draw.text((60, 352), "RAG", font=title_font, fill=ACCENT)

    # Underline accent under "RAG"
    rag_w, _ = text_size(draw, "RAG", title_font)
    draw.rectangle([60, 480, 60 + rag_w + 20, 488], fill=ACCENT)

    # Subtitle
    sub_font = load_font(28, bold=True)
    sub_dim = load_font(24)
    draw.text((60, 510), "ChatGPT-style assistant.", font=sub_font, fill=TEXT_PRIMARY)
    draw.text((60, 548), "Runs 100% on your laptop — no API.", font=sub_dim, fill=TEXT_DIM)

    # Pipeline cards (right side)
    card_font = load_font(28, bold=True)
    card_sub = load_font(20)
    cards = [
        ("WIKIPEDIA", "20 people  ·  20 places"),
        ("CHUNKER", "sentence-aware sliding window"),
        ("CHROMA", "3,668 vectors  ·  cosine"),
        ("OLLAMA", "llama3.2:3b  ·  fully local"),
    ]
    card_x, card_y = 740, 92
    card_w, card_h = 480, 100
    gap = 16
    for title, sub in cards:
        draw.rounded_rectangle(
            [card_x, card_y, card_x + card_w, card_y + card_h],
            radius=16,
            fill=CARD_BG,
            outline=CARD_BORDER,
            width=2,
        )
        # Small accent stripe on the left
        draw.rectangle(
            [card_x + 6, card_y + 18, card_x + 12, card_y + card_h - 18],
            fill=ACCENT,
        )
        draw.text((card_x + 28, card_y + 20), title, font=card_font, fill=TEXT_PRIMARY)
        draw.text((card_x + 28, card_y + 58), sub, font=card_sub, fill=TEXT_DIM)
        card_y += card_h + gap

    # Footer left — author
    footer_bold = load_font(22, bold=True)
    footer_dim = load_font(18)
    draw.text((60, 624), "Mustafa Eren Koç", font=footer_bold, fill=TEXT_PRIMARY)
    draw.text((60, 656), "github.com/mustafaerenkoc44", font=footer_dim, fill=TEXT_DIM)

    # Footer right — DEMO play badge
    badge_x, badge_y = WIDTH - 270, 612
    badge_w, badge_h = 200, 76
    draw.rounded_rectangle(
        [badge_x, badge_y, badge_x + badge_w, badge_y + badge_h],
        radius=18,
        fill=ACCENT,
    )
    # Triangle play icon
    tri = [
        (badge_x + 28, badge_y + 22),
        (badge_x + 28, badge_y + badge_h - 22),
        (badge_x + 60, badge_y + badge_h // 2),
    ]
    draw.polygon(tri, fill=TEXT_PRIMARY)
    badge_font = load_font(34, bold=True)
    draw.text((badge_x + 78, badge_y + 18), "DEMO", font=badge_font, fill=TEXT_PRIMARY)

    img.save(OUTPUT, "PNG", optimize=True)
    print(f"Wrote {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
