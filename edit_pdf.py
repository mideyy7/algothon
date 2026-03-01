"""Edit jump.pdf in-place:
  1. Redact the header text ("ALGOTHON 2026 · JUMP TRADING TRACK" + "Novice Quantitative Engineering")
  2. Redact the footer text ("Page N · Confidential — For Judging Purposes Only")
  3. Insert new pages (after page 4) with code screenshots for
     A — M2vsM1StatArb, B — ETFPackStatArb, C — ETFBasketArb
"""

import keyword
import os
import re
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import fitz  # pymupdf

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = "/Users/Ayomide/Desktop/algothon"
OUT_DIR = f"{BASE}/submission_graphs"
PDF_IN  = f"{BASE}/jump.pdf"

os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Code-snippet strings  (trimmed to fit neatly in a screenshot)
# ═══════════════════════════════════════════════════════════════════════════════

M2VSM1_CODE = '''\
class M2vsM1StatArb:
    """Ridge regression arb: TIDE_SWING ~ b0 + b1*M1 + b2*RV."""

    def step(self):
        m1_mid = self.api.get_mid(self.m1)   # TIDE_SPOT (tidal height)
        m2_mid = self.api.get_mid(self.m2)   # TIDE_SWING (strangle sum)

        # ── Rolling Ridge Regression: M2 ~ b0 + b1*M1 + b2*RV ──────
        rv = self._rv.update(m1_mid)          # realised vol of tides
        self._reg.add(m1_mid, rv, m2_mid);  self._reg.fit()

        fair  = self._reg.predict(m1_mid, rv) # model fair value
        resid = m2_mid - fair                 # residual spread
        z     = (resid - mu) / sd             # normalised z-score
        beta  = self._reg.dy_dx()             # hedge ratio dSWING/dSPOT

        # ── M2 Rich: SELL SWING, hedge long SPOT ──────────────────
        if z > self.z_enter:
            room = max(0, self.max_m2_pos + pos2)
            if room > 0 and bid2 is not None:
                q2    = int(min(self.clip_m2, room))
                self.api.place_order(self.m2, "SELL", bid2, q2)
                hedge = int(np.clip(beta * q2, -self.clip_m1, self.clip_m1))
                if hedge > 0:
                    self.api.place_order(self.m1, "BUY", ask1, min(hedge, room_buy1))

        # ── M2 Cheap: BUY SWING, hedge short SPOT ─────────────────
        if z < -self.z_enter:
            room = max(0, self.max_m2_pos - pos2)
            if room > 0 and ask2 is not None:
                q2    = int(min(self.clip_m2, room))
                self.api.place_order(self.m2, "BUY", ask2, q2)
                hedge = int(np.clip(beta * q2, -self.clip_m1, self.clip_m1))
                if hedge > 0:
                    self.api.place_order(self.m1, "SELL", bid1, min(hedge, room_sell1))
'''

ETFPACK_CODE = '''\
class ETFPackStatArb:
    """Ridge regression arb: LON_FLY fair value ~ f(LON_ETF, RV)."""

    def step(self):
        etf_mid  = self.api.get_mid(self.etf)
        pack_mid = self.api.get_mid(self.pack)

        # ── Rolling Ridge Regression: PACK ~ b0 + b1*ETF + b2*RV ──
        rv = self._rv.update(etf_mid)          # realised vol of ETF
        self._reg.add(etf_mid, rv, pack_mid);  self._reg.fit()

        fair  = self._reg.predict(etf_mid, rv) # model fair value
        r     = pack_mid - fair                # residual spread
        z     = self._resid.z(r)              # normalised z-score
        dP_dE = self._reg.dprice_detf(etf_mid) # delta hedge ratio

        # ── PACK Rich: SELL PACK + hedge ETF long ──────────────────
        if z > self.z_enter:
            room = max(0, self.max_pack_pos + pack_pos)
            if room > 0 and bid_p is not None:
                q_p   = int(min(self.clip_pack, room))
                self.api.place_order(self.pack, "SELL", bid_p, q_p)
                hedge = int(np.clip(dP_dE * q_p, -self.clip_etf, self.clip_etf))
                if hedge > 0:
                    self.api.place_order(self.etf, "BUY", ask_e, min(hedge, room_buy_etf))

        # ── PACK Cheap: BUY PACK + hedge ETF short ─────────────────
        if z < -self.z_enter:
            room = max(0, self.max_pack_pos - pack_pos)
            if room > 0 and ask_p is not None:
                q_p   = int(min(self.clip_pack, room))
                self.api.place_order(self.pack, "BUY", ask_p, q_p)
                hedge = int(np.clip(dP_dE * q_p, -self.clip_etf, self.clip_etf))
                if hedge > 0:
                    self.api.place_order(self.etf, "SELL", bid_e, min(hedge, room_sell_etf))
'''

ETFBASKET_CODE = '''\
class ETFBasketArb:
    """Identity arbitrage: LON_ETF = TIDE_SPOT + WX_SPOT + LHR_COUNT."""

    def step(self):
        etf_mid    = self.api.get_mid(self.etf)
        basket_mid = self._basket_value()   # sum(leg.weight * mid for leg in self.legs)

        # ── Spread = ETF price minus fundamental basket value ─────
        spread = float(etf_mid - basket_mid)

        # ── ETF Rich: SELL ETF, BUY all basket legs ───────────────
        if spread > self.edge:
            qty = int(min(self.clip, etf_room_sell))
            if qty > 0:
                for leg in self.legs:
                    px = self._mid_or_cross_price(leg.product, "BUY")
                    self.api.place_order(leg.product, "BUY", px, qty)
                px_etf = self._mid_or_cross_price(self.etf, "SELL")
                self.api.place_order(self.etf, "SELL", px_etf, qty)

        # ── ETF Cheap: BUY ETF, SELL all basket legs ──────────────
        elif spread < -self.edge:
            qty = int(min(self.clip, etf_room_buy))
            if qty > 0:
                px_etf = self._mid_or_cross_price(self.etf, "BUY")
                self.api.place_order(self.etf, "BUY", px_etf, qty)
                for leg in self.legs:
                    px = self._mid_or_cross_price(leg.product, "SELL")
                    self.api.place_order(leg.product, "SELL", px, qty)
'''


# ═══════════════════════════════════════════════════════════════════════════════
# Code-screenshot renderer  (identical logic to build_report.py)
# ═══════════════════════════════════════════════════════════════════════════════

PAL = {
    "kw":      "#F92672",
    "builtin": "#66D9E8",
    "string":  "#E6DB74",
    "comment": "#75715E",
    "number":  "#AE81FF",
    "self_":   "#FD971F",
    "default": "#F8F8F2",
}
KW_SET = set(keyword.kwlist)


def _tokenise(line: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    i = 0
    while i < len(line):
        if line[i] == "#":
            tokens.append((line[i:], "comment")); break
        m = re.match(r'(["\']).*?\1', line[i:])
        if m:
            tokens.append((m.group(), "string"))
            i += len(m.group()); continue
        m = re.match(r'\b\d+\.?\d*\b', line[i:])
        if m and (i == 0 or not line[i-1].isalnum()):
            tokens.append((m.group(), "number"))
            i += len(m.group()); continue
        m = re.match(r'\b([A-Za-z_]\w*)\b', line[i:])
        if m:
            w = m.group()
            if w in KW_SET:
                kind = "kw"
            elif w in ("True","False","None","int","float","str",
                       "max","min","abs","len","round","print","np"):
                kind = "builtin"
            elif w == "self":
                kind = "self_"
            else:
                kind = "default"
            tokens.append((w, kind)); i += len(w); continue
        tokens.append((line[i], "default")); i += 1
    return tokens


def make_code_screenshot(code: str, path: str) -> None:
    lines     = code.split("\n")
    n_lines   = len(lines)
    char_w    = 0.057
    line_h    = 0.200
    max_chars = max(len(l) for l in lines) + 4
    fig_w     = max(8.5, max_chars * char_w + 0.7)
    fig_h     = n_lines * line_h + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    BG = "#1E1E2E"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    ax.add_patch(mpatches.FancyBboxPatch(
        (0, fig_h - 0.46), fig_w, 0.46,
        boxstyle="square,pad=0", facecolor="#2D2D40", edgecolor="none"))
    for xi, col in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        ax.add_patch(mpatches.Circle(
            (0.22 + xi*0.22, fig_h - 0.22), 0.072, color=col, zorder=3))

    scale = fig_w / (max_chars * char_w + 0.7)
    for li, raw in enumerate(lines):
        y = fig_h - 0.62 - li * line_h
        x = 0.20
        for tok, kind in _tokenise(raw):
            col = PAL.get(kind, PAL["default"])
            ax.text(x, y, tok, fontsize=7.6, color=col,
                    fontfamily="monospace", va="top", transform=ax.transData)
            x += len(tok) * char_w * scale

    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  Code screenshot → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# New-page builder  — matches existing document style (Arial/Helvetica, A4)
# ═══════════════════════════════════════════════════════════════════════════════

# Page size from existing PDF
PW, PH = 595.2, 841.92
MARGIN  = 56.0   # left/right margin (matching existing pages)
TXT_W   = PW - 2 * MARGIN

# Colours matching existing PDF
BLACK  = (0, 0, 0)
DKGREY = (0.20, 0.20, 0.20)
LTGREY = (0.53, 0.53, 0.53)


def _add_strategy_page(
    doc: fitz.Document,
    insert_at: int,
    section_label: str,      # e.g. "C.  Rolling Ridge Regression — M2vsM1StatArb"
    target_line: str,         # e.g. "Target: M1 TIDE_SPOT  ·  M2 TIDE_SWING"
    description: str,         # paragraph body text
    fig_label: str,           # e.g. "Figure C1"
    fig_caption: str,
    code_img_path: str,
) -> None:
    """Insert a new page at insert_at with the strategy section."""
    page = doc.new_page(insert_at, width=PW, height=PH)

    # ── Section heading ────────────────────────────────────────────────────────
    y = 52.0
    page.insert_text(
        (MARGIN, y), section_label,
        fontname="helv", fontsize=12, color=BLACK)
    y += 18

    # Thin rule under heading
    page.draw_line(
        fitz.Point(MARGIN, y), fitz.Point(PW - MARGIN, y),
        color=(0.8, 0.8, 0.8), width=0.6)
    y += 10

    # ── Target line (italic-style, grey) ──────────────────────────────────────
    page.insert_text(
        (MARGIN, y), target_line,
        fontname="helv", fontsize=9, color=LTGREY)
    y += 16

    # ── Description paragraph (word-wrapped) ─────────────────────────────────
    wrapped = textwrap.wrap(description, width=100)
    for line in wrapped:
        page.insert_text((MARGIN, y), line, fontname="helv", fontsize=9.5, color=DKGREY)
        y += 13
    y += 8

    # ── "Engine Implementation" label ────────────────────────────────────────
    page.insert_text(
        (MARGIN, y), "Engine Implementation",
        fontname="hebo", fontsize=9, color=DKGREY)
    y += 12

    # ── Code screenshot image ─────────────────────────────────────────────────
    # Determine image dimensions from file, then scale to fit the text width
    import PIL.Image as PILImage
    with PILImage.open(code_img_path) as im:
        img_w_px, img_h_px = im.size

    available_h = PH - y - 60   # leave room for caption and bottom margin
    img_w_pt = TXT_W
    img_h_pt = img_w_pt * img_h_px / img_w_px
    if img_h_pt > available_h:
        img_h_pt = available_h
        img_w_pt = img_h_pt * img_w_px / img_h_px

    img_rect = fitz.Rect(MARGIN, y, MARGIN + img_w_pt, y + img_h_pt)
    page.insert_image(img_rect, filename=code_img_path)
    y += img_h_pt + 6

    # ── Figure caption ────────────────────────────────────────────────────────
    caption_text = f"{fig_label} — {fig_caption}"
    wrapped_cap = textwrap.wrap(caption_text, width=105)
    for line in wrapped_cap:
        page.insert_text(
            (MARGIN, y), line,
            fontname="helv", fontsize=7.8, color=LTGREY)
        y += 10

    print(f"  ✓  Inserted page at index {insert_at}: {section_label[:50]}")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    # ── Step 1: Generate code screenshots ─────────────────────────────────────
    print("\n── Generating code screenshots ────────────────────────────────")
    p_m2vsm1  = f"{OUT_DIR}/code_m2vsm1.png"
    p_etfpack = f"{OUT_DIR}/code_etfpack.png"
    p_basket  = f"{OUT_DIR}/code_basket.png"

    make_code_screenshot(M2VSM1_CODE,    p_m2vsm1)
    make_code_screenshot(ETFPACK_CODE,   p_etfpack)
    make_code_screenshot(ETFBASKET_CODE, p_basket)

    # ── Step 2: Open PDF and redact header / footer ────────────────────────────
    print("\n── Editing PDF ────────────────────────────────────────────────")
    doc = fitz.open(PDF_IN)

    for page in doc:
        pw, ph = page.rect.width, page.rect.height

        # --- header strip: everything in top 35 pt ---
        top_clip = fitz.Rect(0, 0, pw, 35)
        for blk in page.get_text("dict", clip=top_clip)["blocks"]:
            if blk["type"] != 0:
                continue
            for ln in blk["lines"]:
                for sp in ln["spans"]:
                    page.add_redact_annot(fitz.Rect(sp["bbox"]), fill=(1, 1, 1))

        # --- footer strip: everything in bottom 35 pt ---
        bot_clip = fitz.Rect(0, ph - 35, pw, ph)
        for blk in page.get_text("dict", clip=bot_clip)["blocks"]:
            if blk["type"] != 0:
                continue
            for ln in blk["lines"]:
                for sp in ln["spans"]:
                    page.add_redact_annot(fitz.Rect(sp["bbox"]), fill=(1, 1, 1))

        page.apply_redactions()

    print(f"  ✓  Redacted header/footer on all {len(doc)} pages")

    # ── Step 3: Insert new strategy pages ─────────────────────────────────────
    # Insert after page index 3 (the EWMA graph page), before page index 4
    # (Engineering Safety section). We insert in reverse order so indices stay correct.
    print("\n── Inserting strategy pages ───────────────────────────────────")

    strategies = [
        {
            "section_label": "C.  ETFBasketArb — Identity Arbitrage",
            "target_line":   "Target: M7 LON_ETF  (legs: TIDE_SPOT · WX_SPOT · LHR_COUNT)",
            "description": (
                "LON_ETF settles to exactly TIDE_SPOT + WX_SPOT + LHR_COUNT. "
                "The ETFBasketArb engine monitors the spread between the ETF market price "
                "and the live basket value computed from its three legs. "
                "When the spread exceeds the edge threshold the engine simultaneously "
                "sells the rich side and buys the cheap side across all four products, "
                "locking in a model-free arbitrage profit with zero regression risk."
            ),
            "fig_label":   "Figure C1",
            "fig_caption": (
                "ETFBasketArb.step(): basket value computed from 3 live legs "
                "→ spread vs edge threshold → simultaneous ETF/leg cross execution"
            ),
            "code_img_path": p_basket,
        },
        {
            "section_label": "B.  ETFPackStatArb — Ridge Regression + IV Spread",
            "target_line":   "Target: M7 LON_ETF  ·  M8 LON_FLY",
            "description": (
                "LON_FLY is an exotic option package written on the LON_ETF settlement. "
                "The ETFPackStatArb engine fits a Rolling Ridge Regression "
                "(PACK ~ b0 + b1*ETF + b2*RV) over a 700-tick window to continuously "
                "estimate the option fair value. When the residual Z-score breaches ±2.5σ "
                "the engine sells the rich leg and buys the cheap leg, delta-hedging "
                "the ETF exposure using the regression partial derivative dPACK/dETF."
            ),
            "fig_label":   "Figure B1",
            "fig_caption": (
                "ETFPackStatArb.step(): realised-vol input → Rolling Ridge fit "
                "→ residual Z-score → delta-hedged PACK/ETF execution"
            ),
            "code_img_path": p_etfpack,
        },
        {
            "section_label": "A.  M2vsM1StatArb — Rolling Ridge Regression Z-Score",
            "target_line":   "Target: M1 TIDE_SPOT  ·  M2 TIDE_SWING",
            "description": (
                "TIDE_SWING (Market 2) is a strangle-sum derivative of tidal height "
                "movements, making it structurally related to TIDE_SPOT (Market 1). "
                "The M2vsM1StatArb engine fits a Rolling Ridge Regression "
                "(M2 ~ b0 + b1*M1 + b2*RV) over a 650-tick window. "
                "The residual is normalised to a Z-score over a 350-tick lookback. "
                "Entry at ±2.3σ with a beta-weighted TIDE_SPOT hedge keeps the combined "
                "exposure delta-neutral against tidal movements."
            ),
            "fig_label":   "Figure A1",
            "fig_caption": (
                "M2vsM1StatArb.step(): realised vol → Rolling Ridge fit "
                "→ residual Z-score → β-weighted hedge execution on TIDE_SPOT"
            ),
            "code_img_path": p_m2vsm1,
        },
    ]

    # Insert in reverse order so earlier inserts don't shift later indices
    for strat in strategies:
        _add_strategy_page(
            doc,
            insert_at    = 4,   # always insert at index 4 (reversed order = A, B, C)
            **strat,
        )

    # ── Step 4: Save to temp then replace original ────────────────────────────
    tmp = PDF_IN + ".tmp"
    doc.save(tmp, garbage=4, deflate=True)
    doc.close()
    os.replace(tmp, PDF_IN)
    print(f"\n  ✓  PDF saved → {PDF_IN}")
    print(f"  Total pages: {fitz.open(PDF_IN).page_count}")


if __name__ == "__main__":
    main()
