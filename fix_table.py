"""Fix the broken Live Data Pipeline table in jump.pdf.

The table was split across pages: rows 1-2 on original page 4, rows 3-4 spilled
onto page 8. This script:
  1. Draws the missing rows 3 (Heathrow) and 4 (Derived) at the bottom of the
     table on page 4 (index 3).
  2. Whiteouts the orphaned rows from the top of page 8 (index 7).
"""

import os
import fitz

PDF_PATH = "/Users/Ayomide/Desktop/algothon/jump.pdf"

# ── Exact measurements from existing table (all in PDF user-space points) ──────
# Column x-boundaries (outer left, col-dividers, outer right)
X0  = 56.6   # outer left edge
X1  = 57.4   # inner left (col 1 start)
XD1 = 172.6  # col 1/2 divider left
XD1i= 172.8  # col 1/2 divider right / col 2 start
XD2 = 365.3  # col 2/3 divider left
XD2i= 365.5  # col 2/3 divider right / col 3 start
XD2b= 365.8  # col 3 bg start (slight overshoot in original)
XR  = 538.6  # col 3 bg end
XR1 = 539.3  # outer right edge

# Text x positions
TX1 = 64.1   # col 1 text
TX2 = 179.8  # col 2 text
TX3 = 372.5  # col 3 text

# Font
FONT     = "helv"
FONTSIZE = 8.4

# Text colour: #333333
TXT_COLOR = (0x33/255, 0x33/255, 0x33/255)

# Table border colours
DARK  = (0.5333, 0.5333, 0.5333)  # outer box
MED   = (0.7333, 0.7333, 0.7333)  # inner grid / separators
GREY  = (0.9608, 0.9608, 0.9608)  # alternating row background
WHITE = (1.0, 1.0, 1.0)

# Existing table row 2 ends at (these were measured from the PDF):
# Row 2 bg bottom   : y = 780.2
# Bottom border strip: y = 780.2 → 780.5   (fill=MED, becomes a row separator)
# → Row 3 starts at  : y = 780.5

ROW_H       = 23.5   # height of each data row  (756.7→780.2 = 23.5 pt)
SEP_H       = 0.3    # separator strip height
TEXT_OFFSET = 5.8    # baseline offset inside a row (= 57.4−51.6 from page 8 data)

# ── Row 3 geometry ─────────────────────────────────────────────────────────────
R3_TOP = 780.5
R3_BOT = R3_TOP + ROW_H          # 804.0
R3_TXT = R3_TOP + TEXT_OFFSET    # 786.3

# ── Row 4 geometry ─────────────────────────────────────────────────────────────
R4_SEP_TOP = R3_BOT              # 804.0  (separator line between R3 and R4)
R4_SEP_BOT = R4_SEP_TOP + SEP_H # 804.3
R4_TOP      = R4_SEP_BOT         # 804.3
R4_BOT      = R4_TOP + ROW_H     # 827.8
R4_TXT      = R4_TOP + TEXT_OFFSET # 810.1

# ── Outer bottom border ─────────────────────────────────────────────────────────
BOT_TOP = R4_BOT                  # 827.8
BOT_BOT = BOT_TOP + 0.7          # 828.5


def draw_row_white(page, y_top, y_bot, y_txt, cols):
    """Draw a white (no-fill) data row with borders and text."""
    # Left outer border
    page.draw_rect(fitz.Rect(X0, y_top, X1,  y_bot), fill=DARK,  color=None)
    # Col 1/2 divider
    page.draw_rect(fitz.Rect(XD1, y_top, XD1i, y_bot), fill=MED, color=None)
    # Col 2/3 divider
    page.draw_rect(fitz.Rect(XD2, y_top, XD2i, y_bot), fill=MED, color=None)
    # Right outer border
    page.draw_rect(fitz.Rect(XR,  y_top, XR1,  y_bot), fill=DARK, color=None)
    # Text
    for txt, tx in zip(cols, [TX1, TX2, TX3]):
        page.insert_text((tx, y_txt), txt, fontname=FONT, fontsize=FONTSIZE,
                         color=TXT_COLOR)


def draw_row_grey(page, y_top, y_bot, y_txt, cols):
    """Draw a grey-background data row with borders and text."""
    # Cell backgrounds (grey)
    page.draw_rect(fitz.Rect(X1,   y_top, XD1,  y_bot), fill=GREY, color=None)
    page.draw_rect(fitz.Rect(XD1i, y_top, XD2,  y_bot), fill=GREY, color=None)
    page.draw_rect(fitz.Rect(XD2b, y_top, XR,   y_bot), fill=GREY, color=None)
    # Left outer border
    page.draw_rect(fitz.Rect(X0, y_top, X1,  y_bot), fill=DARK, color=None)
    # Col 1/2 divider
    page.draw_rect(fitz.Rect(XD1, y_top, XD1i, y_bot), fill=MED, color=None)
    # Col 2/3 divider
    page.draw_rect(fitz.Rect(XD2, y_top, XD2i, y_bot), fill=MED, color=None)
    # Right outer border
    page.draw_rect(fitz.Rect(XR,  y_top, XR1,  y_bot), fill=DARK, color=None)
    # Text
    for txt, tx in zip(cols, [TX1, TX2, TX3]):
        page.insert_text((tx, y_txt), txt, fontname=FONT, fontsize=FONTSIZE,
                         color=TXT_COLOR)


def main():
    doc = fitz.open(PDF_PATH)

    # ── Page 4 (index 3): add rows 3 and 4 ─────────────────────────────────────
    page4 = doc[3]

    # Row 3: Heathrow Flights (white row)
    draw_row_white(
        page4, R3_TOP, R3_BOT, R3_TXT,
        ["Heathrow Flights",
         "Heathrow PIHub (official API)",
         "M5 LHR_COUNT, M6 LHR_INDEX"]
    )

    # Separator between row 3 and row 4
    page4.draw_rect(
        fitz.Rect(X0, R4_SEP_TOP, XR1, R4_SEP_BOT),
        fill=MED, color=None)

    # Row 4: Derived (grey row)
    draw_row_grey(
        page4, R4_TOP, R4_BOT, R4_TXT,
        ["Derived",
         "M1 + M3 + M5 computed in-process",
         "M7 LON_ETF, M8 LON_FLY"]
    )

    # Outer bottom border (darker)
    page4.draw_rect(
        fitz.Rect(X0, BOT_TOP, XR1, BOT_BOT),
        fill=DARK, color=None)

    print("  ✓  Added rows 3 & 4 to table on page 4")

    # ── Page 8 (index 7): redact orphaned rows ──────────────────────────────────
    page8 = doc[7]
    # Redact everything from y=28 up to y=100.0 (just before "5. Retrospective" at y=100.1)
    page8.add_redact_annot(fitz.Rect(0, 28, 600, 100), fill=(1, 1, 1))
    page8.apply_redactions(graphics=fitz.PDF_REDACT_LINE_ART_REMOVE_IF_COVERED)
    print("  ✓  Redacted orphaned rows on page 8")

    # ── Save ────────────────────────────────────────────────────────────────────
    tmp = PDF_PATH + ".tmp"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, PDF_PATH)
    print(f"  ✓  Saved → {PDF_PATH}")


if __name__ == "__main__":
    main()
