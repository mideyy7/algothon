"""Build script: generates graphs, code-screenshot images, and the final PDF.

Theme:
  - PDF:   white background, black text throughout
  - Graphs: light (white/off-white) background, coloured data lines
  - Code:  dark background with Monokai syntax colouring (unchanged)
  - Layout: continuous scroll — no forced page breaks between sections
"""

import keyword
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
from PIL import Image

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image as RLImage,
    HRFlowable, Table, TableStyle,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

OUT_DIR  = "/Users/Ayomide/Desktop/algothon/submission_graphs"
PDF_PATH = "/Users/Ayomide/Desktop/algothon/jump.pdf"

W, H = A4

# ── Graph colour palette (designed for light backgrounds) ──────────────────────
C_BLUE   = "#1A6FA8"   # M6 live line
C_ORANGE = "#D4610A"   # OLS fair / EWMA
C_GREEN  = "#1E8A4C"   # buy signals / cheap zone
C_RED    = "#C0392B"   # sell signals / rich zone
C_PURPLE = "#6C3483"   # residual line
C_GREY   = "#888888"   # neutral / grid / zero lines
C_BLACK  = "#1A1A1A"

# ── Light-theme matplotlib defaults (apply to graphs only) ────────────────────
rcParams.update({
    "figure.facecolor":  "white",
    "axes.facecolor":    "#F7F9FC",
    "axes.edgecolor":    "#CCCCCC",
    "axes.labelcolor":   C_BLACK,
    "xtick.color":       C_GREY,
    "ytick.color":       C_GREY,
    "text.color":        C_BLACK,
    "grid.color":        "#E0E4EA",
    "grid.linewidth":    0.7,
    "legend.framealpha": 0.9,
    "legend.edgecolor":  "#CCCCCC",
    "font.family":       "sans-serif",
    "font.size":         9,
})


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 1 — Rolling OLS Co-Integration  (LHR_COUNT vs LHR_INDEX)
# ═══════════════════════════════════════════════════════════════════════════════

def make_graph_ols(path: str) -> None:
    rng = np.random.default_rng(42)
    N   = 480

    m5 = np.cumsum(rng.integers(2, 6, N)).astype(float) + 700
    m5 += rng.normal(0, 8, N)

    beta_true, alpha_true = 0.030, 12.0
    m6_fair = alpha_true + beta_true * m5
    regime  = np.zeros(N)
    regime[180:260] += 22
    m6 = m6_fair + regime + rng.normal(0, 3.5, N)

    window = 120
    ols_fair = np.full(N, np.nan)
    resid    = np.full(N, np.nan)
    for i in range(window, N):
        x = m5[i-window:i]; y = m6[i-window:i]
        mx = x.mean();      my = y.mean()
        vx = ((x-mx)**2).mean() + 1e-9
        b  = ((x-mx)*(y-my)).mean() / vx
        a  = my - b*mx
        ols_fair[i] = a + b*m5[i]
        resid[i]    = m6[i] - ols_fair[i]

    zscore = np.full(N, np.nan)
    zwin   = 80
    for i in range(window+zwin, N):
        r  = resid[i-zwin:i]
        mu = r.mean(); sd = r.std() + 1e-9
        zscore[i] = (resid[i] - mu) / sd

    t = np.arange(N)

    fig = plt.figure(figsize=(13, 7.5), facecolor="white")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08,
                            height_ratios=[3, 2, 2])

    # Panel 1: Price vs OLS Fair
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, m6,       color=C_BLUE,   lw=1.2, alpha=0.90, label="LHR_INDEX (M6) live")
    ax1.plot(t, ols_fair, color=C_ORANGE, lw=2.0, alpha=0.95, label="OLS Fair Value  $\\hat{M6}$", zorder=3)
    ax1.fill_between(t, m6, ols_fair, where=m6 > ols_fair,
                     color=C_RED,   alpha=0.10, label="Rich — sell signal zone")
    ax1.fill_between(t, m6, ols_fair, where=m6 < ols_fair,
                     color=C_GREEN, alpha=0.10, label="Cheap — buy signal zone")
    sell_idx = np.where(zscore > 2.2)[0]
    ax1.scatter(sell_idx, m6[sell_idx], color=C_RED,   marker="v", s=40, zorder=5,
                label="SELL M6 signal", edgecolors=C_BLACK, linewidth=0.3)
    buy_idx  = np.where(zscore < -2.2)[0]
    ax1.scatter(buy_idx,  m6[buy_idx],  color=C_GREEN, marker="^", s=40, zorder=5,
                label="BUY M6 signal",  edgecolors=C_BLACK, linewidth=0.3)
    ax1.set_ylabel("LHR_INDEX (M6)", fontsize=9, color=C_BLACK)
    ax1.legend(fontsize=7.5, ncol=3, loc="upper left")
    ax1.set_title(
        "Rolling OLS Co-Integration Engine — LHR_COUNT (M5) × LHR_INDEX (M6)",
        fontsize=11, color=C_BLACK, pad=8, fontweight="bold")
    ax1.grid(True); ax1.set_xticklabels([])

    # Panel 2: Residual
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, resid, color=C_PURPLE, lw=1.1, alpha=0.9, label="Residual $r_t$")
    ax2.axhline(0, color=C_GREY, lw=0.8, ls="--")
    ax2.set_ylabel("Residual", fontsize=9, color=C_BLACK)
    ax2.legend(fontsize=7.5, loc="upper right")
    ax2.grid(True); ax2.set_xticklabels([])

    # Panel 3: Z-Score
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, zscore, color=C_BLACK, lw=1.0, alpha=0.80, label="Z-score")
    ax3.axhline(+2.2, color=C_RED,   lw=1.2, ls="--", label="+2.2σ SELL entry")
    ax3.axhline(-2.2, color=C_GREEN, lw=1.2, ls="--", label="−2.2σ BUY entry")
    ax3.axhline(+0.6, color=C_GREY,  lw=0.7, ls=":",  label="±0.6σ exit")
    ax3.axhline(-0.6, color=C_GREY,  lw=0.7, ls=":")
    ax3.fill_between(t, zscore, +2.2, where=(zscore > 2.2),  color=C_RED,   alpha=0.15)
    ax3.fill_between(t, zscore, -2.2, where=(zscore < -2.2), color=C_GREEN, alpha=0.15)
    ax3.set_ylabel("Z-Score", fontsize=9, color=C_BLACK)
    ax3.set_xlabel("Tick", fontsize=9, color=C_BLACK)
    ax3.legend(fontsize=7.5, loc="upper right", ncol=2)
    ax3.grid(True)

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  Graph 1 saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# GRAPH 2 — Asymmetric EWMA Dip Buyer  (TIDE_SPOT)
# ═══════════════════════════════════════════════════════════════════════════════

def make_graph_ewma(path: str) -> None:
    rng = np.random.default_rng(7)
    N   = 400

    base  = np.linspace(1380, 1460, N)
    noise = rng.normal(0, 6, N)
    for d in [70, 150, 230, 310, 370]:
        noise[d:d+8] -= rng.uniform(30, 55)
    price = base + noise

    alpha = 0.06
    ewma  = np.zeros(N)
    ewma[0] = price[0]
    for i in range(1, N):
        ewma[i] = (1-alpha)*ewma[i-1] + alpha*price[i]

    dev  = price - ewma
    rstd = np.full(N, np.nan)
    win  = 60
    for i in range(win, N):
        rstd[i] = dev[i-win:i].std() + 1e-9
    zscore   = np.where(~np.isnan(rstd), dev / rstd, np.nan)
    buy_mask = zscore < -1.6
    t = np.arange(N)

    fig = plt.figure(figsize=(13, 7), facecolor="white")
    gs  = gridspec.GridSpec(3, 1, figure=fig, hspace=0.08,
                            height_ratios=[3, 2, 2])

    # Panel 1: Price vs EWMA
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(t, price, color=C_BLUE,   lw=1.1, alpha=0.85, label="TIDE_SPOT live price")
    ax1.plot(t, ewma,  color=C_ORANGE, lw=2.1, alpha=0.95, label="EWMA (α = 0.06)", zorder=3)
    ax1.fill_between(t, price, ewma, where=(price < ewma),
                     color=C_GREEN, alpha=0.10, label="Dip zone")
    ax1.scatter(t[buy_mask], price[buy_mask], color=C_GREEN, marker="^",
                s=55, zorder=6, label="BUY execution",
                edgecolors=C_BLACK, linewidth=0.4)
    ax1.set_ylabel("TIDE_SPOT Price", fontsize=9, color=C_BLACK)
    ax1.set_title("Asymmetric EWMA Dip Buyer Engine — TIDE_SPOT Mean-Reversion",
                  fontsize=11, color=C_BLACK, pad=8, fontweight="bold")
    ax1.legend(fontsize=7.5, ncol=2, loc="upper left")
    ax1.grid(True); ax1.set_xticklabels([])

    # Panel 2: Deviation
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax2.plot(t, dev, color=C_PURPLE, lw=1.1, alpha=0.9)
    ax2.axhline(0, color=C_GREY, lw=0.8, ls="--")
    ax2.fill_between(t, dev, 0, where=(dev < 0), color=C_RED,   alpha=0.12)
    ax2.fill_between(t, dev, 0, where=(dev > 0), color=C_GREEN, alpha=0.08)
    ax2.set_ylabel("Deviation (price − EWMA)", fontsize=9, color=C_BLACK)
    ax2.grid(True); ax2.set_xticklabels([])

    # Panel 3: Z-Score
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax3.plot(t, zscore, color=C_BLACK, lw=1.0, alpha=0.80, label="Z-score (dev / σ)")
    ax3.axhline(-1.6, color=C_GREEN, lw=1.3, ls="--", label="−1.6σ BUY trigger")
    ax3.axhline(-0.3, color=C_GREY,  lw=0.8, ls=":",  label="−0.3σ take-profit exit")
    ax3.fill_between(t, zscore, -1.6,
                     where=~np.isnan(zscore) & (zscore < -1.6),
                     color=C_GREEN, alpha=0.18)
    ax3.scatter(t[buy_mask], zscore[buy_mask], color=C_GREEN, marker="^",
                s=40, zorder=6, edgecolors=C_BLACK, linewidth=0.3)
    ax3.set_ylabel("Z-Score", fontsize=9, color=C_BLACK)
    ax3.set_xlabel("Tick", fontsize=9, color=C_BLACK)
    ax3.legend(fontsize=7.5, loc="upper right")
    ax3.grid(True)

    for ax in (ax1, ax2, ax3):
        ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  ✓  Graph 2 saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CODE SCREENSHOTS  — dark background, Monokai syntax colouring (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════

OLS_CODE = '''\
class PairTradingArb:
    """Rolling OLS co-integration arb: M6 ≈ α + β · M5."""

    def step(self):
        a_mid = self.api.get_mid(self.pA)     # LHR_COUNT live mid
        b_mid = self.api.get_mid(self.pB)     # LHR_INDEX live mid

        # ── Live OLS Fit ──────────────────────────────────────────
        self._ols.add(a_mid, b_mid)
        _a, beta = self._ols.fit()            # closed-form β

        # ── Residual & Z-score ────────────────────────────────────
        fair_b = self._ols.predict(a_mid)
        resid  = b_mid - fair_b
        self._rz.add(resid)
        z = self._rz.z(resid)                # normalised residual

        # ── β-weighted Hedge Sizing ───────────────────────────────
        hedge_a = int(np.clip(round(beta),
                              -self.hedge_clip, self.hedge_clip))

        # ── Asymmetric Execution ──────────────────────────────────
        if z > self.z_enter and room_sell_b > 0:
            q_b = int(min(self.clip_B, room_sell_b))
            self.api.place_order(self.pB, "SELL", bid_b, q_b)
            q_a = int(min(self.clip_A, room_buy_a,
                          abs(hedge_a) * q_b))
            self.api.place_order(self.pA, "BUY",  ask_a, q_a)
'''

EWMA_CODE = '''\
class UpDipBuyer:
    """EWMA mean-reversion — buys sharp dips, never shorts tops."""

    def step(self):
        mid = self.api.get_mid(self.product)
        bbo = self.api.get_best_bid_ask(self.product)

        # ── EWMA tracking & Rolling Volatility ───────────────────
        self._ema.update(mid)                 # α = 0.06
        dev = mid - self._ema.mu
        self._dev.add(dev)
        sd  = self._dev.std()

        # ── Z-score Normalisation ─────────────────────────────────
        z = dev / sd if not np.isnan(sd) else np.nan

        # ── Entry Logic — asymmetric BUY only ────────────────────
        if not np.isnan(z) and z < -self.z_enter:
            pos      = self.api.get_position(self.product)
            room_buy = max(0, self.max_pos - pos)
            if room_buy > 0:
                q = int(min(self.clip, room_buy))
                self.api.place_order(
                    self.product, "BUY", bbo.best_ask, q)

        # ── Take-Profit Exit ──────────────────────────────────────
        if not np.isnan(z) and z > -self.z_take:
            pos = self.api.get_position(self.product)
            if pos > 0:
                q = int(min(self.clip, pos))
                self.api.place_order(
                    self.product, "SELL", bbo.best_bid, q)
'''


def make_code_screenshot(code: str, path: str) -> None:
    """Render Python code as a dark-background PNG with Monokai syntax colouring."""

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

    def tokenise(line: str) -> list[tuple[str, str]]:
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

    lines      = code.split("\n")
    n_lines    = len(lines)
    char_w     = 0.057
    line_h     = 0.200
    max_chars  = max(len(l) for l in lines) + 4
    fig_w      = max(8.5, max_chars * char_w + 0.7)
    fig_h      = n_lines * line_h + 1.0

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    BG = "#1E1E2E"
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, fig_w); ax.set_ylim(0, fig_h); ax.axis("off")

    # Title bar
    ax.add_patch(mpatches.FancyBboxPatch(
        (0, fig_h - 0.46), fig_w, 0.46,
        boxstyle="square,pad=0", facecolor="#2D2D40", edgecolor="none"))
    # Window dots
    for xi, col in enumerate(["#FF5F56", "#FFBD2E", "#27C93F"]):
        ax.add_patch(mpatches.Circle(
            (0.22 + xi*0.22, fig_h - 0.22), 0.072, color=col, zorder=3))

    scale = fig_w / (max_chars * char_w + 0.7)
    for li, raw in enumerate(lines):
        y  = fig_h - 0.62 - li * line_h
        x  = 0.20
        for tok, kind in tokenise(raw):
            col = PAL.get(kind, PAL["default"])
            ax.text(x, y, tok, fontsize=7.6, color=col,
                    fontfamily="monospace", va="top", transform=ax.transData)
            x += len(tok) * char_w * scale

    fig.savefig(path, dpi=180, bbox_inches="tight",
                facecolor=BG, edgecolor="none")
    plt.close(fig)
    print(f"  ✓  Code screenshot saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF BUILDER  — white background, black text, continuous scroll
# ═══════════════════════════════════════════════════════════════════════════════

def build_pdf(
    ols_graph:     str,
    ewma_graph:    str,
    ols_code_img:  str,
    ewma_code_img: str,
    out_path:      str,
) -> None:

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        rightMargin=20*mm, leftMargin=20*mm,
        topMargin=16*mm,   bottomMargin=16*mm,
    )

    # ── Paragraph styles (all black / dark-grey) ──────────────────────────────
    def S(name, **kw):
        return ParagraphStyle(name, parent=getSampleStyleSheet()["Normal"], **kw)

    BLACK   = colors.black
    DKGREY  = colors.HexColor("#333333")
    MDGREY  = colors.HexColor("#555555")
    LTGREY  = colors.HexColor("#888888")
    RULECOL = colors.HexColor("#CCCCCC")
    HDRFILL = colors.HexColor("#2C2C2C")   # table header: near-black
    ROW1    = colors.HexColor("#FFFFFF")
    ROW2    = colors.HexColor("#F5F5F5")
    BRDCOL  = colors.HexColor("#BBBBBB")

    doc_title = S("DocTitle",
        fontName="Helvetica-Bold", fontSize=22, leading=26,
        textColor=BLACK, alignment=TA_CENTER, spaceAfter=2)

    doc_sub = S("DocSub",
        fontName="Helvetica", fontSize=11, leading=14,
        textColor=MDGREY, alignment=TA_CENTER, spaceAfter=1)

    doc_meta = S("DocMeta",
        fontName="Helvetica", fontSize=8.5, leading=11,
        textColor=LTGREY, alignment=TA_CENTER)

    h1 = S("H1",
        fontName="Helvetica-Bold", fontSize=13, leading=16,
        textColor=BLACK, spaceBefore=12, spaceAfter=3)

    h2 = S("H2",
        fontName="Helvetica-Bold", fontSize=10.5, leading=13,
        textColor=DKGREY, spaceBefore=8, spaceAfter=2)

    body = S("Body",
        fontName="Helvetica", fontSize=9.5, leading=14,
        textColor=DKGREY, spaceAfter=6, alignment=TA_JUSTIFY)

    bullet = S("Bullet",
        fontName="Helvetica", fontSize=9.5, leading=13,
        textColor=DKGREY, leftIndent=12, spaceAfter=3)

    caption = S("Caption",
        fontName="Helvetica-Oblique", fontSize=7.8, leading=10,
        textColor=LTGREY, alignment=TA_CENTER, spaceAfter=6)

    label_tag = S("LabelTag",
        fontName="Helvetica-Bold", fontSize=8.5, leading=10,
        textColor=MDGREY, spaceAfter=2)

    quote = S("Quote",
        fontName="Helvetica-BoldOblique", fontSize=11, leading=14,
        textColor=BLACK, alignment=TA_CENTER,
        spaceBefore=8, spaceAfter=8)

    def hr(thickness=0.6, col=RULECOL):
        return HRFlowable(width="100%", thickness=thickness,
                          color=col, spaceAfter=5, spaceBefore=5)

    def sp(h=4): return Spacer(1, h*mm)

    def img(p, width_mm=163):
        iw  = width_mm * mm
        pil = Image.open(p)
        ih  = iw * pil.height / pil.width
        return RLImage(p, width=iw, height=ih)

    def table_style(has_header=True):
        base = [
            ("FONTNAME",    (0,0), (-1,-1),  "Helvetica"),
            ("FONTSIZE",    (0,0), (-1,-1),  8.5),
            ("ROWHEIGHT",   (0,0), (-1,-1),  15),
            ("ALIGN",       (0,0), (-1,-1),  "LEFT"),
            ("LEFTPADDING", (0,0), (-1,-1),  7),
            ("RIGHTPADDING",(0,0), (-1,-1),  5),
            ("GRID",        (0,0), (-1,-1),  0.4, BRDCOL),
            ("BOX",         (0,0), (-1,-1),  0.8, colors.HexColor("#888888")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [ROW1, ROW2]),
            ("TEXTCOLOR",   (0,1), (-1,-1),  DKGREY),
        ]
        if has_header:
            base += [
                ("BACKGROUND", (0,0), (-1,0),  HDRFILL),
                ("TEXTCOLOR",  (0,0), (-1,0),  colors.white),
                ("FONTNAME",   (0,0), (-1,0),  "Helvetica-Bold"),
            ]
        return TableStyle(base)

    CW = W - 40*mm   # usable column width

    elems = []

    # ── Header block (replaces dark cover) ───────────────────────────────────
    elems += [
        sp(4),
        Paragraph("ALGOTHON 2026", doc_title),
        Paragraph("Jump Trading Track — Novice Quantitative Engineering", doc_sub),
        Paragraph("Advanced Quantitative Engineering &amp; Statistical Arbitrage Pipelines", doc_meta),
        Paragraph("March 2026  ·  IMCity Synthetic Exchange", doc_meta),
        sp(3),
        hr(1.5, colors.black),
        sp(2),
    ]

    # ── 1. Executive Summary ──────────────────────────────────────────────────
    elems += [
        Paragraph("1. Executive Summary", h1),
        hr(),
        Paragraph(
            "While most novice teams in this competition approached the IMCity exchange with "
            "hardcoded thresholds or simple moving-average heuristics, our team built a "
            "mathematically rigorous, <b>institutional-grade quantitative trading platform</b> "
            "from the ground up. Every component — from the live data pipeline to the order "
            "execution layer — was engineered with the same precision applied at professional "
            "systematic trading firms.", body),
        Paragraph(
            "We deployed a <b>low-latency event-driven dispatcher</b> that routes every "
            "orderbook tick to one of five proprietary engines, each targeting a distinct "
            "mathematical edge across all eight synthetic London markets. Fair values are "
            "computed dynamically through <b>Rolling Multivariate Ridge Regressions</b>, "
            "<b>Rolling OLS Co-Integration</b>, and <b>Exponentially Weighted Moving "
            "Averages (EWMA)</b> — the bot never statically guesses; it always adapts.", body),
        sp(2),
    ]

    strat_data = [
        ["Strategy", "Markets Targeted", "Mathematical Model"],
        ["A — M2vsM1StatArb",  "TIDE_SPOT, TIDE_SWING",     "Rolling Ridge Regression Z-Score"],
        ["B — ETFPackStatArb", "LON_ETF, LON_FLY",          "Ridge Regression + IV Spread"],
        ["C — ETFBasketArb",   "LON_ETF (all 3 legs)",      "Identity Arbitrage (no model risk)"],
        ["D — UpDipBuyer",     "TIDE_SPOT, WX_SUM, WX_SPOT","Asymmetric EWMA Dip Buyer"],
        ["E — PairTradingArb", "LHR_COUNT, LHR_INDEX",      "Rolling OLS Co-Integration"],
    ]
    cw = [CW*r for r in (0.30, 0.30, 0.40)]
    t1 = Table(strat_data, colWidths=cw)
    t1.setStyle(table_style())
    elems += [t1, sp(5)]

    # ── 2. Advanced Mathematical Models ──────────────────────────────────────
    elems += [
        Paragraph("2. Advanced Mathematical Models", h1),
        hr(),
    ]

    # 2A — OLS Pair Arb
    elems += [
        Paragraph("A.  Rolling OLS Co-Integration Pairs Arbitrage", h2),
        Paragraph("<i>Target: Heathrow Airport Markets — M5: LHR_COUNT  vs  M6: LHR_INDEX</i>",
                  label_tag),
        sp(1),
        Paragraph(
            "Markets 5 and 6 are both derived from exactly the same Heathrow PIHub flight "
            "dataset. M5 counts total movements; M6 measures net directional imbalance. "
            "This creates a <b>latent structural co-integration</b> that our engine "
            "exploits in real time.", body),
        Paragraph(
            "The engine fits a <b>Rolling Ordinary Least Squares</b> regression over a "
            "600-tick window to continuously estimate <i>M6 = α + β(M5)</i>. The residual "
            "is normalised to a Z-score over a 350-tick lookback. When the Z-score "
            "breaches ±2.2σ, the engine executes an <b>asymmetric β-weighted delta hedge</b>: "
            "selling the rich leg and buying the cheap leg in a ratio proportional to the "
            "live OLS beta, keeping the combined position market-neutral.", body),
        sp(1),
    ]

    sig_data = [
        ["Signal Condition", "Action Taken", "Risk Control"],
        ["Z-score > +2.2σ",  "SELL M6 · BUY β×M5",   "clip=12 per leg, max_pos=70"],
        ["Z-score < −2.2σ",  "BUY M6 · SELL β×M5",   "clip=12 per leg, max_pos=70"],
        ["|Z-score| < 0.6σ", "FLATTEN both legs",     "Hard limit ±100 (BotExchangeAdapter)"],
    ]
    cw2 = [CW*r for r in (0.33, 0.35, 0.32)]
    t2  = Table(sig_data, colWidths=cw2)
    t2.setStyle(table_style())
    elems += [t2, sp(3)]

    elems += [
        Paragraph("Engine Implementation", label_tag),
        img(ols_code_img, width_mm=163),
        Paragraph(
            "Figure A1 — PairTradingArb.step(): live OLS fit → residual "
            "→ Z-score → β-weighted hedge execution",
            caption),
        sp(2),
        Paragraph("Visualising the Engine Mechanics", label_tag),
        img(ols_graph, width_mm=163),
        Paragraph(
            "Figure A2 — Simulated LHR_COUNT × LHR_INDEX. Top: live M6 price vs OLS fair "
            "value with signal zones and execution markers. Middle: raw residual series. "
            "Bottom: normalised Z-score with ±2.2σ entry bands and ±0.6σ exit thresholds.",
            caption),
        sp(4),
    ]

    # 2B — EWMA Dip Buyer
    elems += [
        Paragraph("B.  Asymmetric EWMA Dip Buyer", h2),
        Paragraph("<i>Target: TIDE_SPOT  ·  WX_SUM  ·  WX_SPOT</i>", label_tag),
        sp(1),
        Paragraph(
            "Tidal and weather products exhibit structural <b>upward drift</b> — the Thames "
            "approaches High Tide as the settlement window closes, and temperature × humidity "
            "tends to peak mid-day. Symmetric mean-reversion is dangerous here: shorting "
            "artificial tops builds toxic short inventory into a rising market.", body),
        Paragraph(
            "Our engine tracks an <b>Exponentially Weighted Moving Average</b> (α = 0.06) "
            "and computes a rolling Z-score of the deviation. It strictly refuses to short, "
            "acting only on the long side when price dislocates below −1.6σ "
            "(<i>liquidity void / transient flush</i>) and taking profit when the "
            "Z-score recovers above −0.3σ.", body),
        sp(1),
    ]

    param_data = [
        ["Parameter",       "TIDE_SPOT", "WX_SUM", "WX_SPOT"],
        ["EWMA α",          "0.06",      "0.06",   "0.06"],
        ["Entry threshold", "−1.6σ",     "−1.6σ",  "−1.6σ"],
        ["Take-profit",     "−0.3σ",     "−0.3σ",  "−0.3σ"],
        ["Max position",    "±80",       "±45",    "±45"],
        ["Clip per tick",   "12",        "8",      "8"],
    ]
    cw3 = [CW*r for r in (0.30, 0.235, 0.235, 0.23)]
    t3  = Table(param_data, colWidths=cw3)
    t3.setStyle(table_style())
    # Centre numeric cells
    t3.setStyle(TableStyle([("ALIGN", (1,0), (-1,-1), "CENTER")]))
    elems += [t3, sp(3)]

    elems += [
        Paragraph("Engine Implementation", label_tag),
        img(ewma_code_img, width_mm=163),
        Paragraph(
            "Figure B1 — UpDipBuyer.step(): EWMA tracking → Z-score normalisation "
            "→ asymmetric BUY-only entry → take-profit exit",
            caption),
        sp(2),
        Paragraph("Visualising the Engine Mechanics", label_tag),
        img(ewma_graph, width_mm=163),
        Paragraph(
            "Figure B2 — Simulated TIDE_SPOT. Top: live price vs EWMA with buy-execution "
            "markers (▲). Middle: signed deviation from EWMA. Bottom: Z-score with "
            "−1.6σ trigger line and −0.3σ take-profit line.",
            caption),
        sp(4),
    ]

    # ── 3. Engineering Safety ─────────────────────────────────────────────────
    elems += [
        Paragraph("3. Engineering Safety &amp; System Stability", h1),
        hr(),
        Paragraph(
            "Every order placed by any strategy passes first through our centralised "
            "<b>BotExchangeAdapter</b> — a bridge layer that enforces the ±100 competition "
            "position limit as an absolute hard cap before touching the exchange. "
            "No strategy can breach this constraint regardless of signal strength or "
            "simultaneous strategy overlap.", body),
    ]
    for item in [
        ("<b>Sub-Position Clipping:</b> each engine limits its injection per tick "
         "(clip=12 for arb legs, clip=8 for dip buyers), distributing liquidity "
         "gradually rather than wiping out entire orderbook levels."),
        ("<b>Cooldown Timers:</b> all five engines carry independent cooldown periods "
         "(0.15–0.2 s), preventing feedback loops where a partial fill immediately "
         "re-triggers the same signal."),
        ("<b>WARMUP Guards:</b> no engine trades until its statistical estimators "
         "(OLS, Ridge, EWMA) have accumulated sufficient history. NaN propagation "
         "from early ticks is explicitly blocked with early-return guards."),
        ("<b>MM Suppression:</b> passive market-making is disabled on LHR_COUNT and "
         "LHR_INDEX — both legs of the pair arb — eliminating self-churn between "
         "the MM module and PairTradingArb."),
        ("<b>Continuous Integration:</b> 145 unit and integration tests in pytest "
         "validate every strategy, helper class, and dispatch path under mocked "
         "market conditions before any live deployment."),
    ]:
        elems.append(Paragraph(f"• {item}", bullet))
    elems.append(sp(4))

    # ── 4. Live Data Pipeline ─────────────────────────────────────────────────
    elems += [
        Paragraph("4. Live Data Pipeline Engineering", h1),
        hr(),
        Paragraph(
            "Fair-value estimation requires real-world London data ingested faster than "
            "the exchange tick rate. We built a parallel data pipeline fetching three "
            "independent external sources simultaneously on every refresh cycle, with "
            "per-source TTL caching to avoid hammering external APIs.", body),
    ]

    pipe_data = [
        ["Data Source",       "API Endpoint",                    "Markets Priced"],
        ["Thames Water Level","UK Environment Agency (15-min)",  "M1 TIDE_SPOT, M2 TIDE_SWING"],
        ["London Weather",    "Open-Meteo (51.5°N, 0.1°W)",      "M3 WX_SPOT, M4 WX_SUM"],
        ["Heathrow Flights",  "Heathrow PIHub (official API)",   "M5 LHR_COUNT, M6 LHR_INDEX"],
        ["Derived",           "M1 + M3 + M5 computed in-process","M7 LON_ETF, M8 LON_FLY"],
    ]
    cw4 = [CW*r for r in (0.24, 0.40, 0.36)]
    t4  = Table(pipe_data, colWidths=cw4)
    t4.setStyle(table_style())
    elems += [t4, sp(4)]

    # ── 5. Retrospective ─────────────────────────────────────────────────────
    elems += [
        Paragraph("5. Retrospective &amp; Real-World Application", h1),
        hr(),
        Paragraph("Why Institutional Architecture Underperforms in Simulations", h2),
    ]
    for item in [
        ("<b>The Spread-Crossing Tax:</b> Our Ridge and OLS engines correctly identify "
         "structural mispricings but must cross the bid/ask spread to act, instantly "
         "paying a 2-tick transaction cost that eliminates the mathematical edge "
         "in a low-liquidity simulator."),
        ("<b>Synthetic Market Irrationality:</b> Co-integration models rely on markets "
         "eventually reverting to structural laws. In a novice tournament driven by "
         "thousands of random order injections, fundamental relationships can "
         "permanently decouple, turning mathematical edges into drawn-out losses."),
        ("<b>Latency Equality:</b> In this simulator all bots operate on identical "
         "discrete event ticks, neutralising the HFT speed advantage these strategies "
         "depend on in live markets."),
    ]:
        elems.append(Paragraph(f"• {item}", bullet))

    elems += [sp(3), Paragraph("Why This Architecture Wins in Live Markets", h2)]
    for item in [
        ("<b>Making vs Taking Liquidity:</b> At institutional speed, our engines rest "
         "passive limit orders at the calculated fair value, capturing the spread rather "
         "than paying it — reversing the entire cost structure."),
        ("<b>Real-World Co-Integration:</b> Professional market makers enforce structural "
         "bounds across correlated instruments. When LHR_COUNT and LHR_INDEX misprice "
         "relative to their PIHub constraints, multi-billion-dollar firms ruthlessly "
         "arbitrage the gap, confirming the mean-reversion our OLS engine predicts."),
        ("<b>Adaptive Hedging:</b> Hardcoded algorithms blow up when volatility regimes "
         "shift. Our Rolling Ridge and OLS regressions recalculate β tick-by-tick, "
         "keeping the portfolio risk-neutral under macro shocks."),
    ]:
        elems.append(Paragraph(f"• {item}", bullet))

    elems += [
        sp(8),
        hr(1.0, colors.black),
        sp(3),
        Paragraph(
            "We did not build a script to win a game.<br/>"
            "We built an engine to trade a market.",
            quote),
        hr(1.0, colors.black),
    ]

    # ── Page callback: white bg + thin top rule + footer ─────────────────────
    def on_page(canvas, doc):
        canvas.saveState()
        canvas.setFillColor(colors.white)
        canvas.rect(0, 0, W, H, fill=1, stroke=0)
        # Top rule
        canvas.setStrokeColor(colors.black)
        canvas.setLineWidth(1.0)
        canvas.line(20*mm, H - 10*mm, W - 20*mm, H - 10*mm)
        canvas.setFont("Helvetica-Bold", 7.5)
        canvas.setFillColor(colors.black)
        canvas.drawString(20*mm, H - 7.5*mm, "ALGOTHON 2026  ·  JUMP TRADING TRACK")
        canvas.drawRightString(W - 20*mm, H - 7.5*mm, "Novice Quantitative Engineering")
        # Footer
        canvas.setLineWidth(0.4)
        canvas.line(20*mm, 10*mm, W - 20*mm, 10*mm)
        canvas.setFont("Helvetica", 7)
        canvas.setFillColor(colors.HexColor("#888888"))
        canvas.drawCentredString(W/2, 6*mm,
            f"Page {doc.page}  ·  Confidential — For Judging Purposes Only")
        canvas.restoreState()

    doc.build(elems, onFirstPage=on_page, onLaterPages=on_page)
    print(f"\n  ✓  PDF saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n── Generating assets ──────────────────────────────────────────")

    p_ols_graph    = f"{OUT_DIR}/ols_pair_trading.png"
    p_ewma_graph   = f"{OUT_DIR}/ewma_dip_buyer.png"
    p_ols_code     = f"{OUT_DIR}/code_ols.png"
    p_ewma_code    = f"{OUT_DIR}/code_ewma.png"

    make_graph_ols(p_ols_graph)
    make_graph_ewma(p_ewma_graph)
    make_code_screenshot(OLS_CODE,  p_ols_code)
    make_code_screenshot(EWMA_CODE, p_ewma_code)

    print("\n── Building PDF ───────────────────────────────────────────────")
    build_pdf(
        ols_graph     = p_ols_graph,
        ewma_graph    = p_ewma_graph,
        ols_code_img  = p_ols_code,
        ewma_code_img = p_ewma_code,
        out_path      = PDF_PATH,
    )
    print("  Done.\n")
