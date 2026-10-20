"""Build the ten-page development presentation required for the academic portfolio."""

from __future__ import annotations

import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "deliverables" / "Ellinger_Marla_IU14123805_AIUseCase_Hotel_Submission_Dev.pdf"
COMPARISON = ROOT / "docs" / "qwen_comparison.json"
W, H = 960, 540
NAVY = colors.HexColor("#152B36")
TEAL = colors.HexColor("#087E78")
GOLD = colors.HexColor("#DAAB48")
CORAL = colors.HexColor("#C55C42")
CREAM = colors.HexColor("#F7F4EC")
MUTED = colors.HexColor("#52616A")
WHITE = colors.white

FONT_PAIRS = [
    (
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
    ),
    (
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ),
]
for regular, bold in FONT_PAIRS:
    if regular.is_file() and bold.is_file():
        pdfmetrics.registerFont(TTFont("Arial", str(regular)))
        pdfmetrics.registerFont(TTFont("Arial-Bold", str(bold)))
        break
else:
    raise RuntimeError("Install Arial or DejaVu Sans fonts to render the presentation")


def line(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    *,
    size: int = 19,
    color: colors.Color = NAVY,
    bold: bool = False,
) -> None:
    c.setFillColor(color)
    c.setFont("Arial-Bold" if bold else "Arial", size)
    c.drawString(x, y, text)


def wrapped(
    c: canvas.Canvas,
    text: str,
    x: float,
    y: float,
    width: float,
    *,
    size: int = 18,
    leading: int = 25,
    color: colors.Color = NAVY,
) -> float:
    words = text.split()
    parts: list[str] = []
    for word in words:
        attempt = " ".join(parts + [word])
        if parts and pdfmetrics.stringWidth(attempt, "Arial", size) > width:
            line(c, " ".join(parts), x, y, size=size, color=color)
            y -= leading
            parts = [word]
        else:
            parts.append(word)
    if parts:
        line(c, " ".join(parts), x, y, size=size, color=color)
        y -= leading
    return y


def frame(c: canvas.Canvas, number: int, title: str, kicker: str) -> None:
    c.setFillColor(CREAM)
    c.rect(0, 0, W, H, fill=1, stroke=0)
    c.setFillColor(TEAL)
    c.rect(0, H - 14, W, 14, fill=1, stroke=0)
    line(c, kicker.upper(), 54, 477, size=13, color=TEAL, bold=True)
    line(c, title, 54, 426, size=31, bold=True)
    c.setStrokeColor(colors.HexColor("#D6D9D5"))
    c.line(54, 70, 906, 70)
    line(c, "Bellhob · AI Use Case · Development phase", 54, 47, size=12, color=MUTED)
    line(c, f"{number:02d} / 10", 844, 47, size=12, color=MUTED)


def card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    heading: str,
    body: str,
    *,
    accent: colors.Color = TEAL,
) -> None:
    c.setFillColor(WHITE)
    c.roundRect(x, y, w, h, 12, fill=1, stroke=0)
    c.setFillColor(accent)
    c.roundRect(x, y + h - 8, w, 8, 4, fill=1, stroke=0)
    heading_offset, body_offset = (35, 69) if h < 110 else (47, 81)
    line(c, heading, x + 20, y + h - heading_offset, size=20, bold=True)
    wrapped(c, body, x + 20, y + h - body_offset, w - 40, size=15, leading=21, color=MUTED)


def arrow(c: canvas.Canvas, x1: float, y: float, x2: float) -> None:
    c.setStrokeColor(TEAL)
    c.setLineWidth(3)
    c.line(x1, y, x2, y)
    p = c.beginPath()
    p.moveTo(x2, y)
    p.lineTo(x2 - 10, y + 6)
    p.lineTo(x2 - 10, y - 6)
    p.close()
    c.setFillColor(TEAL)
    c.drawPath(p, fill=1, stroke=0)


def source(c: canvas.Canvas, label: str, url: str, x: float, y: float) -> None:
    line(c, label, x, y, size=14, color=TEAL)
    width = pdfmetrics.stringWidth(label, "Arial", 14)
    c.linkURL(url, (x, y - 3, x + width, y + 16), relative=0)


def build() -> Path:
    if not COMPARISON.exists():
        raise FileNotFoundError(f"Run the Qwen comparison before building slides: {COMPARISON}")
    metrics = json.loads(COMPARISON.read_text(encoding="utf-8"))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H))
    c.setTitle("Bellhob - Development phase")
    c.setAuthor("Marla Ellinger")

    frame(c, 1, "Bellhob: hotel booking chatbot", "Development and reflection")
    line(c, "A guarded multilingual academic demo", 54, 343, size=26, color=TEAL)
    wrapped(
        c,
        "Fictional Beit Zeitouna hotel, Beirut. English, German and French room "
        "requests and grounded hotel-policy answers.",
        54,
        287,
        740,
        size=22,
        leading=31,
    )
    line(c, "Marla Ellinger · IU14123805 · 20 October 2026", 54, 153, size=16, color=MUTED)
    c.showPage()

    frame(c, 2, "Problem and acceptance criteria", "Assignment scope")
    card(c, 54, 232, 263, 145, "Collect", "Guest name, check-in, check-out and guest count.")
    card(
        c,
        329,
        232,
        263,
        145,
        "Limit",
        "At most ten user turns. Ask only for missing details.",
        accent=GOLD,
    )
    card(
        c,
        604,
        232,
        302,
        145,
        "Protect",
        "Reject invalid dates and never book before confirmation.",
        accent=CORAL,
    )
    wrapped(
        c,
        "Synthetic rooms, prices and policies keep the exercise reproducible; no payment "
        "or real guest records are handled.",
        54,
        177,
        820,
        size=17,
        leading=25,
    )
    c.showPage()

    frame(c, 3, "Implemented component flow", "Architecture")
    nodes = [
        (55, "Gradio / API"),
        (258, "Qwen proposal"),
        (500, "Validate and route"),
        (742, "Reply + sources"),
    ]
    for x, label in nodes:
        c.setFillColor(WHITE)
        c.roundRect(x, 299, 165 if x != 500 else 185, 68, 10, fill=1, stroke=0)
        line(c, label, x + 14, 327, size=16, bold=True)
    for left, right in ((220, 254), (423, 496), (685, 738)):
        arrow(c, left, 333, right)
    card(
        c,
        96,
        111,
        342,
        129,
        "Knowledge path",
        "Multilingual embeddings → FAISS → authored policy and source ID.",
    )
    card(
        c,
        523,
        111,
        342,
        129,
        "Booking path",
        "Typed state → search → quote → explicit confirmation → SQLite.",
    )
    c.showPage()

    frame(c, 4, "Booking tools have a fixed order", "Safety mechanism")
    steps = [
        "1  Search rooms",
        "2  Calculate quote",
        "3  Confirm exact details",
        "4  Create reservation",
    ]
    for i, text in enumerate(steps):
        y = 346 - i * 67
        c.setFillColor(WHITE)
        c.roundRect(75, y - 25, 510, 52, 9, fill=1, stroke=0)
        line(c, text, 95, y - 3, size=20, bold=True)
        if i < 3:
            c.setStrokeColor(TEAL)
            c.line(330, y - 26, 330, y - 41)
    card(
        c,
        619,
        124,
        266,
        225,
        "Guarded state",
        "Qwen suggests actions. Code checks required "
        "slots, valid dates, inventory and confirmation before any database write.",
        accent=CORAL,
    )
    c.showPage()

    frame(c, 5, "Training data and provenance", "Data collection")
    card(
        c,
        54,
        215,
        402,
        165,
        "Intent examples",
        "102 synthetic messages; six intents; "
        "66 train, 18 development, 18 test. English, German and French.",
    )
    card(
        c,
        478,
        215,
        428,
        165,
        "BIO slot examples",
        "10 synthetic annotated utterances; "
        "six train, two development, two test. Entity suggestions remain advisory.",
        accent=GOLD,
    )
    wrapped(
        c,
        "No real guest messages were collected. The fixed split and duplicate check "
        "support reproducible experiments, not real-world performance claims.",
        54,
        164,
        830,
        size=17,
        leading=25,
    )
    c.showPage()

    frame(c, 6, "Model selection used development data", "NLP training")
    bars = [
        ("Word TF-IDF", 0.771, MUTED),
        ("Character TF-IDF", 0.716, GOLD),
        ("Multilingual encoder", 0.828, TEAL),
    ]
    for i, (label, value, color) in enumerate(bars):
        y = 350 - i * 80
        line(c, label, 57, y + 10, size=18)
        c.setFillColor(colors.HexColor("#E5E9E5"))
        c.roundRect(280, y, 500, 31, 8, fill=1, stroke=0)
        c.setFillColor(color)
        c.roundRect(280, y, 500 * value, 31, 8, fill=1, stroke=0)
        line(c, f"{value:.3f}", 800, y + 8, size=16, bold=True)
    wrapped(
        c,
        "Selected intent model: 15/18 test correct (83%). BIO token F1: 0.632 on "
        "only two test utterances.",
        57,
        125,
        820,
        size=18,
        leading=26,
    )
    c.showPage()

    frame(c, 7, "Policy answers are sourced", "RAG pipeline")
    labels = ["Question", "Embed", "FAISS search", "Threshold", "Authored answer"]
    xpos = [58, 225, 390, 572, 745]
    for x, label in zip(xpos, labels, strict=True):
        c.setFillColor(WHITE)
        c.roundRect(x, 285, 150, 72, 9, fill=1, stroke=0)
        line(c, label, x + 13, 315, size=16, bold=True)
    for left, right in ((208, 221), (375, 386), (557, 568), (722, 741)):
        arrow(c, left, 321, right)
    line(c, "17 / 20", 95, 186, size=34, color=TEAL, bold=True)
    line(c, "correct source first", 95, 156, size=18, color=MUTED)
    line(c, "2 / 2", 535, 186, size=34, color=TEAL, bold=True)
    line(c, "unsupported questions rejected", 535, 156, size=18, color=MUTED)
    c.showPage()

    frame(c, 8, "A reservation needs confirmation", "Dialogue example")
    card(
        c,
        54,
        294,
        830,
        83,
        "Guest",
        "Book a double room 2030-10-01 to 2030-10-02 for 2 guests. My name is Alex Kim.",
    )
    card(
        c,
        54,
        187,
        830,
        83,
        "Assistant",
        "Room available. Demo price shown. Reply exactly: confirm booking.",
        accent=GOLD,
    )
    card(
        c,
        54,
        80,
        830,
        83,
        "Guest → assistant",
        "confirm booking → reservation reference created.",
        accent=TEAL,
    )
    c.showPage()

    frame(c, 9, "Comparison on 30 synthetic dialogues", "Evaluation")
    guarded = metrics["guarded"]
    baseline = metrics["baseline"]
    card(
        c,
        54,
        210,
        400,
        178,
        "Guarded Qwen",
        f"{guarded['passed']}/30 expected routes "
        f"and effects. {guarded['unsafe_preconfirmation_writes']} database writes "
        "before confirmation.",
    )
    card(
        c,
        478,
        210,
        428,
        178,
        "Single-prompt baseline",
        f"{baseline['unsafe_preconfirmation_claims']} claimed bookings before confirmation; "
        f"{baseline['completed_valid_bookings']}/6 valid bookings claimed after confirmation.",
        accent=CORAL,
    )
    wrapped(
        c,
        "Baseline 'bookings' are model claims only. The baseline has no database access. "
        "Both evaluations use authored prompts, so results are descriptive.",
        54,
        160,
        830,
        size=16,
        leading=24,
    )
    c.showPage()

    frame(c, 10, "Reproduction and limitations", "Reflection")
    wrapped(
        c,
        "Run the app locally or with Docker. Retrain from repository data, then run "
        "retrieval and dialogue scripts. The Qwen model is pinned and downloaded separately.",
        54,
        360,
        850,
        size=19,
        leading=29,
    )
    wrapped(
        c,
        "Next iteration: independent guest-like annotations, entity-level evaluation, "
        "threshold calibration, and more policy paraphrases. This fictional demo does not "
        "process real payments or guest records.",
        54,
        244,
        850,
        size=19,
        leading=29,
    )
    source(c, "Python", "https://www.python.org/", 54, 120)
    source(c, "Pydantic", "https://docs.pydantic.dev/", 158, 120)
    source(c, "Gradio", "https://www.gradio.app/", 286, 120)
    source(c, "FAISS", "https://github.com/facebookresearch/faiss", 396, 120)
    source(c, "Sentence-Transformers", "https://www.sbert.net/", 495, 120)
    source(c, "Qwen model", "https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct", 712, 120)
    c.showPage()

    c.save()
    return OUTPUT


if __name__ == "__main__":
    print(build())
