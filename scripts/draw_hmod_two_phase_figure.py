"""Draw the corrected two-phase H-MOD paper figure.

Phase 1 follows the code path in hmod/training.py:
  objective library -> basic low-level skills + advanced high-level skills ->
  skill-conditioned low policy / skill library.

Phase 2 follows train_hmod.py and train_hmod_2agent.py:
  a macro goal conditions the meta-controller; self-play feedback adapts the
  controller through hint / experience memories while the frozen low-level
  policy executes the selected local objective weight.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "images"

BLUE = "#2F6FCC"
RED = "#E63B3B"
GREEN = "#2F9E44"
ORANGE = "#F97316"
PURPLE = "#7C3AED"
TEAL = "#0891B2"
TEXT = "#101828"
MUTED = "#475467"


def box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle="",
    *,
    fc="#F8FBFF",
    ec=BLUE,
    dashed=True,
    lw=2.1,
    title_size=13,
    sub_size=9,
    title_color=TEXT,
    z=3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=z,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(x + w / 2, y + h * 0.66, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color, zorder=z + 1)
        ax.text(x + w / 2, y + h * 0.34, subtitle, ha="center", va="center",
                fontsize=sub_size, color=TEXT, linespacing=1.18, zorder=z + 1)
    else:
        ax.text(x + w / 2, y + h / 2, title, ha="center", va="center",
                fontsize=title_size, fontweight="bold", color=title_color,
                linespacing=1.05, zorder=z + 1)
    return patch


def empty_box(ax, x, y, w, h, *, fc="#F8FBFF", ec=BLUE, dashed=True, lw=2.1, z=3):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.02,rounding_size=0.16",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=z,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax,
    x1,
    y1,
    x2,
    y2,
    *,
    color=BLUE,
    label=None,
    rad=0.0,
    dashed=False,
    lw=2.0,
    ms=14,
    text_offset=(0, 0),
    z=8,
):
    a = FancyArrowPatch(
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
    ax.add_patch(a)
    if label:
        ax.text((x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1],
                label, ha="center", va="center", fontsize=8.8, color=color,
                fontstyle="italic", zorder=z + 1)
    return a


def simplex(ax, x, y, s=1.0):
    pts = [(x + 0.10 * s, y + 0.05 * s), (x + 0.74 * s, y + 0.95 * s), (x + 1.38 * s, y + 0.05 * s)]
    ax.add_patch(Polygon(pts, closed=True, facecolor="white", edgecolor="#60A5FA", linewidth=1.3, zorder=5))
    for px, py, c in [(x + 0.50 * s, y + 0.36 * s, RED), (x + 0.89 * s, y + 0.34 * s, GREEN), (x + 0.73 * s, y + 0.62 * s, BLUE)]:
        ax.add_patch(Circle((px, py), 0.045 * s, facecolor=c, edgecolor="white", linewidth=0.7, zorder=6))
    ax.text(x + 0.74 * s, y - 0.10 * s, "[sl_ratio, fairness, deal_rate]", fontsize=7.6,
            color=MUTED, ha="center", zorder=6)


def dialogue_icon(ax, x, y, s=1.0):
    for yy, c in [(0.52, "#93C5FD"), (0.28, "#FDBA74"), (0.04, "#93C5FD")]:
        ax.add_patch(FancyBboxPatch((x + 0.38 * s, y + yy * s), 1.06 * s, 0.17 * s,
                                    boxstyle="round,pad=0.02,rounding_size=0.04",
                                    facecolor=c, edgecolor="none", zorder=5))
    for xx, yy, c in [(x + 0.10 * s, y + 0.57 * s, "#BBF7D0"), (x + 1.60 * s, y + 0.33 * s, "#FBCFE8")]:
        ax.add_patch(Circle((xx, yy), 0.075 * s, facecolor="#FFD7B5", edgecolor=TEXT, linewidth=0.5, zorder=6))
        ax.add_patch(Rectangle((xx - 0.07 * s, yy - 0.20 * s), 0.14 * s, 0.14 * s,
                               facecolor=c, edgecolor=TEXT, linewidth=0.5, zorder=6))


def tiny_network(ax, x, y, s=1.0):
    # encoder
    ax.add_patch(FancyBboxPatch((x, y), 0.82 * s, 0.42 * s,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#FFF7CC", edgecolor="#667085", linewidth=1.0, zorder=5))
    for i in range(3):
        ax.plot([x + 0.10 * s, x + 0.43 * s], [y + (0.12 + 0.10 * i) * s, y + (0.09 + 0.10 * i) * s],
                color="#9CA3AF", linewidth=1.0, zorder=6)
    ax.text(x + 0.41 * s, y - 0.13 * s, "RoBERTa", fontsize=7.5, fontweight="bold", ha="center")
    arrow(ax, x + 0.88 * s, y + 0.21 * s, x + 1.16 * s, y + 0.21 * s, color=BLUE, lw=1.6, ms=9)
    layers = [
        [(x + 1.28 * s, y + 0.10 * s), (x + 1.28 * s, y + 0.22 * s), (x + 1.28 * s, y + 0.34 * s)],
        [(x + 1.52 * s, y + 0.06 * s), (x + 1.52 * s, y + 0.16 * s), (x + 1.52 * s, y + 0.28 * s), (x + 1.52 * s, y + 0.38 * s)],
        [(x + 1.76 * s, y + 0.11 * s), (x + 1.76 * s, y + 0.22 * s), (x + 1.76 * s, y + 0.33 * s)],
    ]
    for l1, l2 in zip(layers, layers[1:]):
        for p in l1:
            for q in l2:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=TEXT, linewidth=0.35, alpha=0.6, zorder=5)
    colors = ["#F87171", "#FDE047", "#60A5FA", "#34D399"]
    for layer in layers:
        for i, p in enumerate(layer):
            ax.add_patch(Circle(p, 0.032 * s, facecolor=colors[i % 4], edgecolor=TEXT, linewidth=0.3, zorder=6))
    ax.text(x + 1.52 * s, y - 0.13 * s, "DQN", fontsize=7.5, fontweight="bold", ha="center")


def skill_stack(ax, x, y, s=1.0, colors=("#DDFBFF", "#D9F99D"), label=""):
    for i in range(4):
        ax.add_patch(FancyBboxPatch((x + i * 0.08 * s, y - i * 0.06 * s), 0.86 * s, 0.30 * s,
                                    boxstyle="round,pad=0.015,rounding_size=0.05",
                                    facecolor=colors[i % 2], edgecolor="#2563EB",
                                    linestyle="--", linewidth=1.0, zorder=5 + i))
    if label:
        ax.text(x + 0.58 * s, y - 0.42 * s, label, fontsize=7.6, ha="center", color=MUTED, zorder=9)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 7.2), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    # Background phases.
    ax.add_patch(Rectangle((0.35, 0.48), 8.55, 6.30, facecolor="#EAF6E6", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((8.90, 0.48), 8.80, 6.30, facecolor="#FFF1C7", edgecolor="none", zorder=0))
    ax.plot([8.90, 8.90], [0.48, 6.78], color="#D5C791", linewidth=1.2, zorder=1)

    ax.text(0.55, 6.42, "Phase 1", fontsize=26, color=RED, fontweight="bold", ha="left")
    ax.text(2.22, 6.42, "Train low-level and high-level skills", fontsize=16.5, color=TEXT, fontweight="bold", ha="left")
    ax.text(9.15, 6.42, "Phase 2", fontsize=26, color=RED, fontweight="bold", ha="left")
    ax.text(10.82, 6.42, "Macro-goal meta-controller adaptation", fontsize=16.5, color=TEXT, fontweight="bold", ha="left")

    # Phase 1 left inputs.
    box(ax, 0.65, 4.45, 1.85, 1.35, "Dialogue Data", "state, action,\nreward signals", fc="#F8FBFF", title_size=12.5)
    dialogue_icon(ax, 0.82, 4.65, s=0.72)
    box(ax, 0.65, 1.55, 1.85, 1.45, "Objective\nLibrary", "buyer intents\nand weight rules", fc="#FFFDF2", title_size=12)
    simplex(ax, 1.02, 1.78, s=0.72)

    empty_box(ax, 2.95, 4.18, 2.10, 1.18, fc="#E0F2FE")
    ax.text(4.00, 5.08, "Low-Level\nSkill Builder", fontsize=12.5, fontweight="bold",
            ha="center", va="center", linespacing=0.9, zorder=8)
    ax.text(4.00, 4.57, "basic skills\nfixed objective weights", fontsize=8.7,
            ha="center", va="center", zorder=8)
    ax.text(4.00, 4.28, r"$z_i^{low}\rightarrow w_i$", fontsize=9.2, color=BLUE,
            fontweight="bold", ha="center", zorder=8)

    empty_box(ax, 2.95, 2.18, 2.10, 1.18, fc="#ECFDF3")
    ax.text(4.00, 3.08, "High-Level\nSkill Builder", fontsize=12.5, fontweight="bold",
            ha="center", va="center", linespacing=0.9, zorder=8)
    ax.text(4.00, 2.57, "advanced skills\nmacro clusters", fontsize=8.7,
            ha="center", va="center", zorder=8)
    ax.text(4.00, 2.28, r"$z_j^{high}\rightarrow \bar w_j$", fontsize=9.2, color=GREEN,
            fontweight="bold", ha="center", zorder=8)

    empty_box(ax, 5.65, 4.10, 2.75, 1.42, fc="#F8FBFF")
    ax.text(7.03, 5.22, "Skill-Conditioned Low Policy", fontsize=12.8,
            fontweight="bold", ha="center", zorder=8)
    ax.text(7.03, 5.00, "learns actions for each skill weight", fontsize=8.8,
            color=MUTED, ha="center", zorder=8)
    tiny_network(ax, 5.95, 4.42, s=0.76)
    ax.text(7.82, 4.61, r"$Q(s,a,w)$", fontsize=10.4, color=RED, fontweight="bold", ha="center", zorder=8)

    box(ax, 5.65, 1.75, 2.75, 1.28, "Hierarchical\nSkill Library", r"$\mathcal{Z}=\mathcal{Z}_{low}\cup\mathcal{Z}_{high}$", fc="#FFF7ED", ec=ORANGE, title_size=12.8, sub_size=11)
    skill_stack(ax, 6.65, 2.14, s=0.82, colors=("#DDFBFF", "#DCFCE7"), label="trained reusable skills")

    box(ax, 3.15, 0.72, 2.90, 0.70, "Multi-objective dialogue environment", "rollout under sampled skill weight", fc="#F8FBFF", title_size=11.2, sub_size=8.4)
    box(ax, 6.55, 0.72, 1.95, 0.70, "Checkpoint", "frozen low policy", fc="#D1FAE5", ec=GREEN, dashed=False, title_size=11.2, sub_size=8.6)

    # Phase 1 arrows.
    arrow(ax, 2.50, 5.10, 2.95, 4.86, color=BLUE)
    arrow(ax, 2.50, 2.28, 2.95, 4.55, color=BLUE, rad=-0.12)
    arrow(ax, 2.50, 2.28, 2.95, 2.80, color=BLUE)
    arrow(ax, 5.05, 4.77, 5.65, 4.77, color=BLUE)
    arrow(ax, 5.05, 2.76, 5.65, 2.38, color=BLUE)
    arrow(ax, 6.82, 4.10, 4.60, 1.42, color=BLUE, rad=0.12, label="rollout", text_offset=(-0.22, 0.05))
    arrow(ax, 5.80, 1.42, 6.55, 1.08, color=RED, label="TD / GPI", text_offset=(-0.05, 0.14))
    arrow(ax, 7.22, 4.10, 7.22, 3.03, color=RED, label="skill reuse", text_offset=(0.48, 0.00))
    arrow(ax, 7.20, 1.75, 7.28, 1.42, color=BLUE)

    # Phase 2 macro-goal adaptation.
    box(ax, 9.35, 5.18, 1.60, 0.82, "Macro Goal", r"$g$: ambiguous buyer goal", fc="#DDFBFF", title_size=12.5, sub_size=9.0)
    box(ax, 11.10, 5.18, 1.75, 0.82, "Dialogue", r"$h_t$: visible history", fc="#F8FBFF", title_size=12.5, sub_size=9.0)
    dialogue_icon(ax, 11.38, 5.30, s=0.48)
    box(ax, 13.00, 5.18, 1.55, 0.82, "Seller Intent", r"$I_t$: drift state", fc="#F3E8FF", title_size=12.2, sub_size=9.0)
    box(ax, 14.72, 5.18, 1.72, 0.82, "Hints / Exp.", "adapted memory", fc="#D9F99D", title_size=12.2, sub_size=9.0)

    box(ax, 9.55, 3.72, 3.05, 1.10, "Meta-Controller", "maps macro goal to a\nlocal high-level skill / weight", fc="#F8FBFF", title_size=14, sub_size=9.3)
    ax.text(10.18, 3.86, r"$g,h_t,I_t,H_t$", fontsize=10, color=PURPLE, fontweight="bold", ha="center", zorder=8)
    ax.text(11.55, 3.86, r"$\rightarrow z_t^{high}, w_t$", fontsize=10, color=RED, fontweight="bold", ha="center", zorder=8)

    box(ax, 13.05, 3.72, 1.70, 1.10, "Intent-Drift\nDetector", "drift? current intent", fc="#F3E8FF", title_size=11.8, sub_size=8.5)
    box(ax, 14.95, 3.72, 1.70, 1.10, "High-Policy\nAdapter", "allocation to w_t", fc="#F3E8FF", title_size=11.8, sub_size=8.5)

    box(ax, 9.65, 2.30, 2.28, 0.92, "Selected Skill", r"$z_t^{high}$ or $w_t=[p,f,d]$", fc="#FFFDF2", ec=ORANGE, title_size=12.5, sub_size=10)
    box(ax, 12.45, 2.18, 3.05, 1.05, "Frozen Low-Level Policy", r"$a_t=\arg\max_a w_t^TQ(h_t,a,w_t)$", fc="#F8FBFF", title_size=13.2, sub_size=9.6)
    tiny_network(ax, 12.75, 2.50, s=0.68)
    ax.text(15.05, 2.54, r"$Q(h,a,w_t)$", fontsize=10.0, color=RED, fontweight="bold", ha="center", zorder=8)

    box(ax, 9.80, 1.05, 1.60, 0.80, "Action", "strategy + price", fc="#E0F2FE", ec=GREEN, dashed=False, title_size=12)
    box(ax, 11.75, 1.05, 1.60, 0.80, "Safety Mask", "price ceiling", fc="#FEE2E2", ec=RED, dashed=False, title_size=12)
    box(ax, 13.70, 1.05, 1.70, 0.80, "Dynamic Seller", "intent drift", fc="#FFFDF2", ec=ORANGE, dashed=False, title_size=12)
    box(ax, 15.78, 1.05, 1.52, 0.80, "Metrics", "GSR / T2DA / CVR", fc="#E0F2FE", title_size=12, sub_size=8.8)

    box(ax, 15.62, 2.22, 1.68, 0.72, "Meta Adaptation", "review hints + exp.", fc="#FEF3C7", ec=RED, dashed=True, title_size=10.8, sub_size=8.2)

    # Phase 2 arrows.
    for x1, x2 in [(10.15, 10.45), (11.98, 11.20), (13.78, 13.90), (15.58, 12.15)]:
        arrow(ax, x1, 5.18, x2, 4.82, color=BLUE, lw=1.7, ms=10)
    arrow(ax, 12.60, 4.28, 13.05, 4.28, color=BLUE, label="two-agent option", text_offset=(0.00, 0.16), lw=1.6, ms=10)
    arrow(ax, 14.75, 4.28, 14.95, 4.28, color=BLUE, label="on drift", text_offset=(0.0, 0.16), lw=1.6, ms=10)
    arrow(ax, 15.80, 3.72, 11.78, 3.22, color=BLUE, rad=0.08, label="w_t", text_offset=(0.15, 0.15), lw=1.7, ms=10)
    arrow(ax, 11.08, 3.72, 11.08, 3.22, color=BLUE, lw=2.0)
    arrow(ax, 11.93, 2.76, 12.45, 2.76, color=BLUE)
    arrow(ax, 13.38, 2.18, 10.60, 1.85, color=BLUE, rad=0.08)
    arrow(ax, 11.40, 1.45, 11.75, 1.45, color=BLUE)
    arrow(ax, 13.35, 1.45, 13.70, 1.45, color=BLUE)
    arrow(ax, 15.40, 1.45, 15.78, 1.45, color=BLUE)

    # Self-play feedback loop routed around the right edge.
    ax.plot([16.55, 17.50], [1.85, 1.85], color=RED, linestyle="--", linewidth=1.6, zorder=4)
    ax.plot([17.50, 17.50], [1.85, 5.60], color=RED, linestyle="--", linewidth=1.6, zorder=4)
    arrow(ax, 17.50, 5.60, 16.44, 5.60, color=RED, dashed=True, lw=1.6, ms=10, label="feedback", text_offset=(0.10, 0.18))
    arrow(ax, 16.15, 1.85, 16.05, 2.22, color=RED, dashed=True, lw=1.6, ms=9)
    ax.plot([16.90, 17.32], [2.58, 2.58], color=RED, linewidth=1.5, linestyle="--", zorder=4)
    ax.plot([17.32, 17.32], [2.58, 3.50], color=RED, linewidth=1.5, linestyle="--", zorder=4)
    ax.plot([17.32, 12.30], [3.50, 3.50], color=RED, linewidth=1.5, linestyle="--", zorder=4)
    arrow(ax, 12.30, 3.50, 12.10, 3.72, color=RED, dashed=True, rad=0.02, lw=1.5, ms=10,
          label="adapt meta-controller", text_offset=(1.45, -0.12))

    # Phase link.
    arrow(ax, 8.50, 1.07, 9.65, 2.76, color=GREEN, lw=2.4, ms=15,
          label="frozen skills", text_offset=(0.20, 0.12))
    arrow(ax, 8.42, 2.38, 9.65, 2.76, color=GREEN, lw=1.9, ms=12)

    fig.tight_layout(pad=0.0)
    for ext in ("jpg", "png", "pdf", "svg"):
        out = OUT_DIR / f"hmod_two_phase_figure.{ext}"
        if ext == "jpg":
            fig.savefig(out, dpi=300, pil_kwargs={"quality": 95, "subsampling": 0})
        else:
            fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
