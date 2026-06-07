"""Draw a paper-style H-MOD architecture figure.

The figure is intentionally generated with matplotlib primitives instead of
Graphviz so the layout resembles an architecture illustration suitable for a
paper: grouped panels, dashed modules, colored execution/optimization arrows,
small schematic icons, and compact equations.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Polygon,
    Rectangle,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"


BLUE = "#2F6FCC"
LIGHT_BLUE = "#E7F2FF"
RED = "#E63737"
GREEN_BG = "#E9F5E3"
YELLOW_BG = "#FFF2C7"
BOX_EDGE = "#4B89D8"
TEXT = "#111827"
MUTED = "#4B5563"
GREEN = "#3FA34D"
ORANGE = "#F97316"
PURPLE = "#7C3AED"


def add_round_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle="",
    fc="white",
    ec=BOX_EDGE,
    dashed=True,
    lw=2.0,
    title_size=12,
    sub_size=9,
    title_color=TEXT,
    z=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
        zorder=z,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.63,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight="bold",
            zorder=z + 1,
        )
        ax.text(
            x + w / 2,
            y + h * 0.33,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=TEXT,
            linespacing=1.2,
            zorder=z + 1,
        )
    else:
        ax.text(
            x + w / 2,
            y + h / 2,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            color=title_color,
            fontweight="bold",
            linespacing=1.1,
            zorder=z + 1,
        )
    return patch


def add_empty_box(
    ax,
    x,
    y,
    w,
    h,
    fc="white",
    ec=BOX_EDGE,
    dashed=True,
    lw=2.0,
    z=2,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
        linestyle="--" if dashed else "-",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def add_arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    color=BLUE,
    label=None,
    rad=0.0,
    dashed=False,
    lw=2.0,
    ms=14,
    text_offset=(0, 0),
    z=4,
):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=ms,
        linewidth=lw,
        color=color,
        linestyle="--" if dashed else "-",
        connectionstyle=f"arc3,rad={rad}",
        zorder=z,
    )
    ax.add_patch(arrow)
    if label:
        ax.text(
            (x1 + x2) / 2 + text_offset[0],
            (y1 + y2) / 2 + text_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=8.5,
            color=color,
            fontstyle="italic",
            zorder=z + 1,
        )
    return arrow


def draw_dialogue_icon(ax, x, y, scale=1.0):
    # Two tiny users and three utterance bars.
    colors = ["#60A5FA", "#FDBA74", "#93C5FD"]
    for i, yy in enumerate([y + 0.52 * scale, y + 0.24 * scale, y - 0.04 * scale]):
        ax.add_patch(
            FancyBboxPatch(
                (x + 0.35 * scale, yy),
                1.08 * scale,
                0.18 * scale,
                boxstyle="round,pad=0.02,rounding_size=0.04",
                facecolor=colors[i],
                edgecolor="none",
                zorder=3,
            )
        )
    for xx, yy, c in [(x + 0.07 * scale, y + 0.58 * scale, "#A7F3D0"), (x + 1.58 * scale, y + 0.32 * scale, "#FBCFE8")]:
        ax.add_patch(Circle((xx, yy), 0.08 * scale, facecolor="#F8D6B3", edgecolor=TEXT, linewidth=0.6, zorder=4))
        ax.add_patch(Rectangle((xx - 0.08 * scale, yy - 0.21 * scale), 0.16 * scale, 0.16 * scale, facecolor=c, edgecolor=TEXT, linewidth=0.5, zorder=4))


def draw_roberta_dqn(ax, x, y, scale=1.0, frozen=False):
    # Paper/encoder icon.
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            1.28 * scale,
            0.62 * scale,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            facecolor="#FFF8CC",
            edgecolor="#6B7280",
            linewidth=1.2,
            zorder=3,
        )
    )
    for i in range(3):
        ax.plot(
            [x + 0.12 * scale, x + 0.52 * scale],
            [y + (0.17 + 0.13 * i) * scale, y + (0.12 + 0.12 * i) * scale],
            color="#9CA3AF",
            linewidth=1.4,
            zorder=4,
        )
    ax.add_patch(Circle((x + 0.92 * scale, y + 0.36 * scale), 0.10 * scale, facecolor="#FDE68A", edgecolor="none", zorder=4))
    ax.add_patch(Circle((x + 1.10 * scale, y + 0.36 * scale), 0.10 * scale, facecolor="#FDE68A", edgecolor="none", zorder=4))
    ax.text(x + 0.66 * scale, y - 0.17 * scale, "RoBERTa", fontsize=8.5, fontweight="bold", ha="center", zorder=4)
    if frozen:
        ax.text(x + 1.18 * scale, y + 0.66 * scale, "frozen", fontsize=7.2, color=BLUE, fontweight="bold", ha="right", zorder=4)

    # Arrow to neural net.
    add_arrow(ax, x + 1.38 * scale, y + 0.31 * scale, x + 1.78 * scale, y + 0.31 * scale, color=BLUE, lw=1.8, ms=10)

    # Tiny DQN graph.
    layers = [
        [(x + 1.95 * scale, y + 0.15 * scale), (x + 1.95 * scale, y + 0.31 * scale), (x + 1.95 * scale, y + 0.47 * scale)],
        [(x + 2.25 * scale, y + 0.08 * scale), (x + 2.25 * scale, y + 0.24 * scale), (x + 2.25 * scale, y + 0.40 * scale), (x + 2.25 * scale, y + 0.56 * scale)],
        [(x + 2.55 * scale, y + 0.16 * scale), (x + 2.55 * scale, y + 0.31 * scale), (x + 2.55 * scale, y + 0.46 * scale)],
    ]
    for left, right in zip(layers, layers[1:]):
        for p in left:
            for q in right:
                ax.plot([p[0], q[0]], [p[1], q[1]], color="#111827", linewidth=0.45, alpha=0.65, zorder=3)
    node_colors = ["#F87171", "#FDE047", "#60A5FA", "#34D399"]
    for layer in layers:
        for j, p in enumerate(layer):
            ax.add_patch(Circle(p, 0.04 * scale, facecolor=node_colors[j % len(node_colors)], edgecolor=TEXT, linewidth=0.4, zorder=4))
    ax.text(x + 2.25 * scale, y - 0.17 * scale, "DQN", fontsize=8.5, fontweight="bold", ha="center", zorder=4)


def draw_buffer_icon(ax, x, y, scale=1.0):
    # Server stack + database cylinder.
    for i in range(3):
        ax.add_patch(Rectangle((x, y + i * 0.18 * scale), 0.72 * scale, 0.13 * scale, facecolor="#60A5FA", edgecolor=TEXT, linewidth=0.7, zorder=4))
        ax.add_patch(Circle((x + 0.08 * scale, y + i * 0.18 * scale + 0.065 * scale), 0.025 * scale, facecolor="#111827", zorder=5))
    ax.add_patch(Rectangle((x + 0.86 * scale, y + 0.07 * scale), 0.45 * scale, 0.38 * scale, facecolor="#FACC15", edgecolor=TEXT, linewidth=0.8, zorder=4))
    ax.add_patch(Circle((x + 1.085 * scale, y + 0.45 * scale), 0.225 * scale, facecolor="#FDE68A", edgecolor=TEXT, linewidth=0.8, zorder=5))


def draw_simplex(ax, x, y, scale=1.0):
    pts = [(x + 0.1 * scale, y + 0.05 * scale), (x + 0.75 * scale, y + 0.95 * scale), (x + 1.35 * scale, y + 0.05 * scale)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="#F8FAFC", edgecolor="#60A5FA", linewidth=1.2, zorder=4))
    ax.plot([pts[0][0], pts[1][0]], [pts[0][1], pts[1][1]], color="#CBD5E1", linewidth=0.8)
    ax.plot([pts[1][0], pts[2][0]], [pts[1][1], pts[2][1]], color="#CBD5E1", linewidth=0.8)
    ax.plot([pts[2][0], pts[0][0]], [pts[2][1], pts[0][1]], color="#CBD5E1", linewidth=0.8)
    for px, py, c in [(x + 0.55 * scale, y + 0.38 * scale, RED), (x + 0.9 * scale, y + 0.35 * scale, GREEN), (x + 0.75 * scale, y + 0.63 * scale, BLUE)]:
        ax.add_patch(Circle((px, py), 0.04 * scale, facecolor=c, edgecolor="white", linewidth=0.5, zorder=5))
    ax.text(x + 0.75 * scale, y - 0.12 * scale, "Preference simplex", fontsize=7.5, ha="center", color=MUTED, zorder=5)


def draw_robot(ax, x, y, scale=1.0):
    ax.add_patch(FancyBboxPatch((x, y), 0.55 * scale, 0.42 * scale, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#E0F2FE", edgecolor=TEXT, linewidth=0.8, zorder=4))
    ax.add_patch(Circle((x + 0.18 * scale, y + 0.25 * scale), 0.035 * scale, facecolor=BLUE, zorder=5))
    ax.add_patch(Circle((x + 0.37 * scale, y + 0.25 * scale), 0.035 * scale, facecolor=BLUE, zorder=5))
    ax.plot([x + 0.19 * scale, x + 0.36 * scale], [y + 0.13 * scale, y + 0.13 * scale], color=TEXT, linewidth=0.7, zorder=5)
    ax.plot([x + 0.27 * scale, x + 0.27 * scale], [y + 0.42 * scale, y + 0.53 * scale], color=TEXT, linewidth=0.7, zorder=4)
    ax.add_patch(Circle((x + 0.27 * scale, y + 0.56 * scale), 0.035 * scale, facecolor=ORANGE, zorder=5))


def draw_sliders(ax, x, y, scale=1.0):
    for yy, knob in [(0.45, 0.55), (0.28, 0.25), (0.11, 0.72)]:
        ax.plot([x, x + 0.82 * scale], [y + yy * scale, y + yy * scale], color="#06B6D4", linewidth=2.0, zorder=4)
        ax.add_patch(Circle((x + knob * scale, y + yy * scale), 0.045 * scale, facecolor="white", edgecolor="#06B6D4", linewidth=1.5, zorder=5))


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 7.4), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.4)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Background panels.
    ax.add_patch(Rectangle((0.35, 0.45), 11.55, 6.45, facecolor=GREEN_BG, edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((11.9, 0.45), 5.75, 6.45, facecolor=YELLOW_BG, edgecolor="none", zorder=0))
    ax.plot([11.9, 11.9], [0.45, 6.9], color="#D6CFA5", linewidth=1.0, zorder=1)

    ax.text(0.55, 6.55, "Training", fontsize=28, fontweight="bold", color=RED, ha="left", va="center")
    ax.text(11.98, 6.55, "Inference", fontsize=28, fontweight="bold", color=RED, ha="left", va="center")

    # Legend.
    add_arrow(ax, 8.45, 6.57, 9.25, 6.57, color=BLUE, lw=2.2, ms=12)
    ax.text(9.35, 6.57, "execution", fontsize=10, color=BLUE, va="center")
    add_arrow(ax, 10.05, 6.57, 10.85, 6.57, color=RED, lw=2.2, ms=12)
    ax.text(10.95, 6.57, "optimization", fontsize=10, color=RED, va="center")

    # Training modules.
    add_round_box(ax, 0.65, 4.25, 2.20, 1.80, "Dialogue Cases", "history, item context,\nmacro-goal", fc="#F8FBFF", title_size=14)
    draw_dialogue_icon(ax, 0.92, 4.55, scale=1.0)

    add_empty_box(ax, 0.65, 1.05, 2.20, 1.75, fc="#FFFDF2")
    ax.text(1.75, 2.46, "Buyer Objectives", fontsize=14, fontweight="bold", ha="center", va="center")
    draw_simplex(ax, 1.02, 1.36, scale=0.90)
    ax.text(1.75, 1.18, "static w, stages, rules", fontsize=8.5, ha="center", color=MUTED)

    add_round_box(ax, 3.25, 4.55, 2.00, 1.40, "H-MOD\nScenario Gen.", "constraints + persona\nintent drift trigger", fc="#F0FFF4", title_size=13)
    add_empty_box(ax, 5.55, 4.55, 2.95, 1.40, fc="#F8FBFF")
    ax.text(7.02, 5.72, "R-PADPP Low-Policy", fontsize=14, fontweight="bold", ha="center")
    ax.text(7.02, 5.49, "anchor curriculum + regret-gated GPI", fontsize=8.8, ha="center", color=MUTED)
    draw_roberta_dqn(ax, 5.82, 4.78, scale=0.58)
    ax.text(7.95, 4.95, r"$Q(s,a,w)$", fontsize=11, color=RED, fontweight="bold", ha="center")

    add_empty_box(ax, 8.95, 4.55, 1.65, 1.40, fc="#F8FBFF")
    ax.text(9.78, 5.72, "Replay\nBuffer", fontsize=13, fontweight="bold", ha="center", va="top", linespacing=0.9)
    draw_buffer_icon(ax, 9.22, 4.82, scale=0.62)

    add_empty_box(ax, 10.82, 4.55, 1.48, 1.40, fc="#FFFDF2")
    ax.text(11.56, 5.74, "Regret\nGate", fontsize=13, fontweight="bold", ha="center", va="top", linespacing=0.9)
    ax.text(11.56, 5.08, r"$W_{conv}$", fontsize=11, color=MUTED, ha="center")
    ax.text(11.56, 4.82, r"$Reg(w)<\epsilon$", fontsize=9.0, color=RED, fontweight="bold", ha="center")

    add_empty_box(ax, 8.95, 2.50, 1.65, 1.45, fc="#F8FBFF")
    ax.text(9.78, 3.70, "Pref. Space", fontsize=13, fontweight="bold", ha="center")
    draw_simplex(ax, 9.13, 2.78, scale=0.72)

    add_empty_box(ax, 3.25, 1.15, 3.25, 1.85, fc="#F8FBFF")
    ax.text(4.88, 2.66, "Multi-objective Dialogue Env.", fontsize=14, fontweight="bold", ha="center")
    ax.text(4.88, 2.42, "buyer agent + dynamic seller simulator", fontsize=8.8, color=MUTED, ha="center")
    draw_robot(ax, 3.55, 1.70, scale=1.0)
    ax.add_patch(FancyBboxPatch((4.35, 1.72), 0.80, 0.48, boxstyle="round,pad=0.02,rounding_size=0.08", facecolor="#DBEAFE", edgecolor="#38BDF8", linewidth=1.2, zorder=4))
    ax.plot([5.35, 5.48, 5.61, 5.74, 5.87], [1.75, 2.16, 1.75, 2.16, 1.75], color="#2563EB", linewidth=2.2, zorder=4)
    ax.text(5.95, 1.72, "drift", fontsize=9, color=MUTED, va="center")

    add_round_box(ax, 6.90, 1.40, 1.95, 1.18, "Transition", r"$(s,a,r,s')$", fc="#FFFDF2", ec=GREEN, dashed=False, title_size=13, sub_size=11)

    add_round_box(ax, 9.55, 1.20, 2.15, 1.35, "GPI / TD Update", r"$L=(1-\alpha)L_{self}+\alpha L_{know}$", fc="#FFFDF2", ec=ORANGE, dashed=True, title_size=13, sub_size=10)
    add_round_box(ax, 6.72, 0.58, 2.30, 0.65, "LLM Hint Training", "metrics -> reusable playbook", fc="#F3E8FF", ec=PURPLE, dashed=True, title_size=11, sub_size=8)

    # Training arrows.
    add_arrow(ax, 2.85, 5.15, 3.25, 5.15, color=BLUE)
    add_arrow(ax, 2.85, 1.90, 3.25, 4.75, color=BLUE, rad=-0.10)
    add_arrow(ax, 5.25, 5.15, 5.55, 5.15, color=BLUE)
    add_arrow(ax, 7.02, 4.55, 5.30, 3.00, color=BLUE, rad=0.10, label="policy action", text_offset=(-0.10, 0.12))
    add_arrow(ax, 6.50, 2.12, 6.90, 2.02, color=BLUE)
    add_arrow(ax, 7.88, 2.58, 9.26, 4.55, color=BLUE, rad=0.12)
    add_arrow(ax, 10.60, 5.15, 10.82, 5.15, color=RED)
    add_arrow(ax, 9.78, 3.95, 9.78, 2.55, color=RED, label="sample w", text_offset=(0.45, 0.08))
    add_arrow(ax, 8.85, 1.98, 9.55, 1.92, color=RED)
    add_arrow(ax, 10.62, 2.55, 11.10, 4.55, color=RED, rad=-0.15)
    add_arrow(ax, 9.55, 1.38, 8.50, 4.55, color=RED, rad=-0.14, label="update Q", text_offset=(-0.20, -0.08))
    add_arrow(ax, 6.50, 1.18, 7.50, 0.90, color=RED, dashed=True, lw=1.6, ms=10)

    # Inference modules.
    add_round_box(ax, 12.25, 5.68, 1.35, 0.75, "Macro Goal", r"$g$", fc="#DDFBFF", title_size=12, sub_size=10)
    add_round_box(ax, 13.78, 5.68, 1.68, 0.75, "Dial. History", r"$h_t$", fc="#F8FBFF", title_size=12, sub_size=10)
    add_round_box(ax, 15.66, 5.68, 1.35, 0.75, "Hints / Exp.", r"$H_t$", fc="#D9F99D", title_size=12, sub_size=10)
    draw_dialogue_icon(ax, 14.04, 5.76, scale=0.42)

    add_round_box(ax, 12.40, 4.35, 4.95, 1.08, "H-MOD Meta-Controller", "Intent-Drift Detector + High-Policy LLM", fc="#F8FBFF", title_size=15, sub_size=10)
    add_round_box(ax, 12.70, 3.45, 1.92, 0.66, "Detector", "drift? intent", fc="#F3E8FF", title_size=11, sub_size=8)
    add_round_box(ax, 15.00, 3.45, 1.92, 0.66, "High Policy", "NL allocation to w_t", fc="#F3E8FF", title_size=11, sub_size=8)
    add_arrow(ax, 14.62, 3.78, 15.00, 3.78, color=BLUE, label="on drift", text_offset=(0, 0.16), lw=1.6, ms=10)

    add_empty_box(ax, 12.40, 2.25, 4.95, 1.00, fc="#F8FBFF")
    ax.text(14.88, 3.05, "Frozen R-PADPP Planner", fontsize=14, fontweight="bold", ha="center")
    draw_roberta_dqn(ax, 12.70, 2.47, scale=0.52, frozen=True)
    ax.text(15.58, 2.70, r"$a_t=\arg\max_a\, w_t^T Q(h_t,a,w_t)$", fontsize=11.2, color=TEXT, ha="center", zorder=5)
    ax.text(16.65, 2.44, r"$Q(h,a,w_t)$", fontsize=11, color=RED, fontweight="bold", ha="center", zorder=5)

    add_round_box(ax, 12.50, 1.15, 1.85, 0.82, "Buyer Action", "strategy + price", fc="#E0F2FE", ec=GREEN, dashed=False, title_size=12, sub_size=9)
    add_round_box(ax, 14.55, 1.15, 1.75, 0.82, "Safety Mask", "ceiling check", fc="#FEE2E2", ec=RED, dashed=False, title_size=12, sub_size=9)
    add_round_box(ax, 16.45, 1.15, 1.18, 0.82, "Seller", "drifts", fc="#FFFDF2", ec=ORANGE, dashed=False, title_size=12, sub_size=9)
    add_round_box(ax, 16.25, 0.40, 1.35, 0.58, "GSR / T2DA / CVR", "", fc="#E0F2FE", ec=BOX_EDGE, dashed=True, title_size=9.6)

    # Inference arrows.
    add_arrow(ax, 12.92, 5.68, 13.15, 5.43, color=BLUE, lw=1.6, ms=10)
    add_arrow(ax, 14.62, 5.68, 14.70, 5.43, color=BLUE, lw=1.6, ms=10)
    add_arrow(ax, 16.34, 5.68, 16.05, 5.43, color=BLUE, lw=1.6, ms=10)
    add_arrow(ax, 14.88, 4.35, 14.88, 4.11, color=BLUE, lw=1.8, ms=10)
    add_arrow(ax, 13.66, 3.45, 13.25, 3.05, color=BLUE, rad=0.12, lw=1.6, ms=10)
    add_arrow(ax, 15.96, 3.45, 15.82, 3.43, color=BLUE, lw=1.4, ms=8)
    add_arrow(ax, 14.88, 4.35, 14.88, 3.43, color=BLUE, lw=1.8, ms=10)
    add_arrow(ax, 14.88, 3.45, 14.88, 3.43, color=BLUE, lw=1.6, ms=8)
    add_arrow(ax, 14.88, 3.43, 14.88, 3.25, color=BLUE, lw=2.0, ms=12, label=r"$w_t=[p,f,d]$", text_offset=(0.68, 0.12))
    add_arrow(ax, 14.88, 2.25, 13.43, 1.97, color=BLUE, lw=2.0)
    add_arrow(ax, 14.35, 1.56, 14.55, 1.56, color=BLUE, lw=1.8, ms=10)
    add_arrow(ax, 16.30, 1.56, 16.45, 1.56, color=BLUE, lw=1.8, ms=10)
    ax.plot([17.05, 17.72], [1.97, 1.97], color=BLUE, linewidth=1.5, linestyle="--", zorder=3)
    ax.plot([17.72, 17.72], [1.97, 6.58], color=BLUE, linewidth=1.5, linestyle="--", zorder=3)
    ax.plot([17.72, 14.62], [6.58, 6.58], color=BLUE, linewidth=1.5, linestyle="--", zorder=3)
    add_arrow(ax, 14.62, 6.58, 14.62, 6.43, color=BLUE, dashed=True, lw=1.5, ms=9, label="new history", text_offset=(1.05, 0.16))
    add_arrow(ax, 15.42, 1.15, 16.25, 0.74, color=RED, dashed=True, lw=1.3, ms=9, label="CVR", text_offset=(-0.05, -0.08))
    add_arrow(ax, 17.05, 1.15, 16.94, 0.98, color=RED, dashed=True, lw=1.4, ms=9)
    ax.plot([17.60, 17.82], [0.69, 0.69], color=RED, linewidth=1.4, linestyle="--", zorder=3)
    ax.plot([17.82, 17.82], [0.69, 6.05], color=RED, linewidth=1.4, linestyle="--", zorder=3)
    add_arrow(ax, 17.82, 6.05, 17.02, 6.05, color=RED, dashed=True, lw=1.4, ms=9, label="feedback", text_offset=(0.02, 0.18))

    # Cross-panel connection.
    add_arrow(ax, 11.20, 4.92, 12.40, 2.76, color=GREEN, lw=2.2, ms=14, label="frozen low policy", text_offset=(0.20, 0.12))

    fig.tight_layout(pad=0.0)
    for ext in ("jpg", "png", "pdf", "svg"):
        out = OUT_DIR / f"hmod_paper_figure.{ext}"
        if ext == "jpg":
            fig.savefig(out, dpi=300, pil_kwargs={"quality": 95, "subsampling": 0})
        else:
            fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
