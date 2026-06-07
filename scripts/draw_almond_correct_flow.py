"""Draw a clean, corrected ALMOND flow figure.

The figure follows the finalized algorithm logic:
1) Phase 1 learns the reusable low-level policy theta* and skill library Z.
2) Phase 2 freezes theta*, runs self-play, and updates hints H^(e) -> H^(e+1).
3) Inference loads theta*, Z, H*, M_phi and performs no updates.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper"

BLUE = "#1459D9"
RED = "#E3342F"
GREEN = "#148A2E"
PURPLE = "#5B21B6"
ORANGE = "#F97316"
TEXT = "#111827"
MUTED = "#4B5563"


def rounded_box(
    ax,
    x,
    y,
    w,
    h,
    title,
    subtitle="",
    *,
    fc="white",
    ec=BLUE,
    dashed=True,
    lw=1.6,
    title_size=10.5,
    sub_size=8.0,
    color=TEXT,
    z=3,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.08",
        facecolor=fc,
        edgecolor=ec,
        linewidth=lw,
        linestyle="--" if dashed else "-",
        zorder=z,
    )
    ax.add_patch(patch)
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.64,
            title,
            ha="center",
            va="center",
            fontsize=title_size,
            fontweight="bold",
            color=color,
            linespacing=1.0,
            zorder=z + 1,
        )
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=sub_size,
            color=TEXT,
            linespacing=1.15,
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
            fontweight="bold",
            color=color,
            linespacing=1.0,
            zorder=z + 1,
        )
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
    dashed=False,
    rad=0.0,
    lw=1.7,
    ms=12,
    text_offset=(0.0, 0.0),
    z=8,
):
    patch = FancyArrowPatch(
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
    ax.add_patch(patch)
    if label:
        ax.text(
            (x1 + x2) / 2 + text_offset[0],
            (y1 + y2) / 2 + text_offset[1],
            label,
            ha="center",
            va="center",
            fontsize=7.3,
            color=color,
            fontstyle="italic",
            zorder=z + 1,
        )
    return patch


def icon_circle(ax, x, y, label, color=BLUE):
    ax.add_patch(Circle((x, y), 0.15, facecolor="#F8FAFC", edgecolor=color, linewidth=1.2, zorder=6))
    ax.text(x, y, label, ha="center", va="center", fontsize=10, fontweight="bold", color=color, zorder=7)


def panel(ax, x, y, w, h, title, subtitle, edge, fc):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.03,rounding_size=0.09",
            facecolor=fc,
            edgecolor=edge,
            linewidth=1.4,
            zorder=1,
        )
    )
    ax.text(x + w / 2, y + h - 0.35, title, ha="center", va="center",
            fontsize=13, color=edge, fontweight="bold", zorder=2)
    ax.text(x + w / 2, y + h - 0.63, subtitle, ha="center", va="center",
            fontsize=9.2, color=edge, fontstyle="italic", zorder=2)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(18, 10), dpi=300)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 10)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    ax.text(
        9,
        9.72,
        "ALMOND: Hierarchical Dialogue Policy Learning",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="#030712",
    )
    ax.text(
        9,
        9.38,
        "Adaptive Language-guided Multi-objective Dialogue with Intent-Drift Monitoring",
        ha="center",
        va="center",
        fontsize=13,
        color="#111827",
        fontstyle="italic",
    )

    # Panels.
    panel(
        ax,
        0.15,
        0.95,
        5.35,
        8.1,
        "Training Phase 1: Low-Level Skill Learning",
        "learn low-level skill policy only",
        BLUE,
        "#F6FBFF",
    )
    panel(
        ax,
        5.65,
        0.95,
        7.1,
        8.1,
        "Training Phase 2: High-Level Meta-Controller Adaptation",
        "freeze low-level policy, run self-play, and update hints only",
        PURPLE,
        "#FCF8FF",
    )
    panel(
        ax,
        12.95,
        0.95,
        4.9,
        8.1,
        "Inference",
        "use learned components only (no updates)",
        ORANGE,
        "#FFF9EF",
    )

    # Phase 1.
    rounded_box(ax, 0.45, 7.05, 2.15, 0.95, "Training Data &\nObjective Library",
                "dialogue cases,\nmacro objectives,\ntraining scenarios", fc="#FFFFFF")
    icon_circle(ax, 0.78, 7.55, "D", BLUE)
    rounded_box(ax, 2.85, 7.05, 2.15, 0.95, "Scenario &\nIntent-Drift Generator",
                "personas, constraints,\ndrift triggers", fc="#FFFFFF")
    icon_circle(ax, 3.18, 7.55, "S", GREEN)
    rounded_box(ax, 0.95, 5.85, 3.75, 0.88, r"Hierarchical Skill Library $\mathcal{Z}$",
                "low-level and high-level\ncommunication skills", fc="#FFFFFF")
    icon_circle(ax, 1.25, 6.29, "Z", PURPLE)
    rounded_box(ax, 0.95, 4.65, 3.75, 0.82, "Multi-objective Dialogue Environment",
                "simulated interaction trajectories", fc="#FFFFFF")
    icon_circle(ax, 1.25, 5.06, "E", BLUE)
    rounded_box(ax, 0.75, 3.23, 4.15, 1.0, "Regret-Gated Low-Level Skill Learning",
                r"skill-conditioned rollouts + $Q(s,a,w)$ update", fc="#FFF7F7", ec=RED)
    icon_circle(ax, 1.08, 3.74, "Q", RED)
    rounded_box(ax, 0.95, 2.0, 3.75, 0.82, r"Frozen Low-Level Policy $\theta^*$",
                "reusable skill executor", fc="#FFFFFF")
    icon_circle(ax, 1.25, 2.41, "*", BLUE)
    rounded_box(ax, 0.55, 1.25, 4.55, 0.45, r"Outputs of Phase 1: $\theta^*, \mathcal{Z}$",
                "", fc="#FFFFFF", title_size=12)

    arrow(ax, 2.6, 7.55, 2.85, 7.55)
    arrow(ax, 1.52, 7.05, 1.52, 6.73)
    arrow(ax, 3.92, 7.05, 3.92, 6.73)
    arrow(ax, 2.82, 5.85, 2.82, 5.47)
    arrow(ax, 2.82, 4.65, 2.82, 4.23, label="rollouts", text_offset=(0.5, 0.04))
    arrow(ax, 4.0, 5.85, 4.0, 4.23, color=BLUE, rad=0.0, lw=1.2,
          label=r"condition on $\mathcal{Z}$", text_offset=(0.55, -0.12))
    arrow(ax, 2.82, 3.23, 2.82, 2.82, color=RED)
    arrow(ax, 2.82, 2.0, 2.82, 1.7)

    # Phase 2.
    input_y = 7.35
    in_w = 1.25
    x_inputs = [5.95, 7.45, 8.95, 10.45]
    labels = [
        ("Macro Goal", r"$g$"),
        ("Dialogue\nHistory", r"$h_t$"),
        ("Constraints", r"$c_t$"),
        ("Current\nHints", r"$\mathcal{H}^{(e)}$"),
    ]
    for x, (t, s) in zip(x_inputs, labels):
        rounded_box(ax, x, input_y, in_w, 0.75, t, s, fc="#FFFFFF", ec=PURPLE, title_size=8.5, sub_size=8.2)

    rounded_box(ax, 6.05, 6.42, 4.25, 0.58, r"1. Intent-Drift Detector $\mathcal{D}_{\psi}$",
                r"consume $h_t$ and infer $(\hat{i}_t,\delta_t)$", fc="#FFFFFF", ec=PURPLE, dashed=False,
                title_size=9.2, sub_size=7.6)
    rounded_box(ax, 6.05, 5.68, 4.25, 0.58, r"2. Hint-Conditioned Meta-Controller $\mathcal{M}_{\phi}(\cdot;\mathcal{H}^{(e)})$",
                "high-level planning / guidance", fc="#FFFFFF", ec=PURPLE, dashed=False,
                title_size=8.7, sub_size=7.6)
    rounded_box(ax, 6.05, 4.94, 4.25, 0.58, r"3. Skill / Weight Selection $z_t / w_t$",
                r"select skill or local objective weight over $\mathcal{Z}$", fc="#FFFFFF", ec=PURPLE, dashed=False,
                title_size=9.2, sub_size=7.4)
    rounded_box(ax, 5.95, 2.92, 3.65, 1.72, r"4. Self-Play Rollout with Frozen $\theta^*$",
                "simulated interaction", fc="#FFFFFF", ec=BLUE, dashed=True,
                title_size=9.0, sub_size=7.4)
    rounded_box(ax, 6.35, 3.86, 2.85, 0.34, r"Dialogue Action $a_t$", "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=7.8)
    rounded_box(ax, 6.35, 3.42, 2.85, 0.34, "Safety & Constraint Checker", "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=7.8)
    rounded_box(ax, 6.35, 2.98, 2.85, 0.34, "User / Opponent Response", "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=7.8)

    rounded_box(ax, 10.55, 2.45, 1.82, 2.95, "Hint Distillation Loop",
                "", fc="#FFFFFF", ec=RED, dashed=True, title_size=9.5)
    rounded_box(ax, 10.85, 4.82, 1.22, 0.42, r"Input: $\mathcal{H}^{(e)}$",
                "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=8.2)
    rounded_box(ax, 10.85, 4.20, 1.22, 0.45, "Metrics\nGSR, T2DA, CVR",
                "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=7.6)
    rounded_box(ax, 10.85, 3.45, 1.22, 0.52, "Strategic Hint\nDistillation",
                "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=7.6)
    rounded_box(ax, 10.85, 2.72, 1.22, 0.48, r"$\mathcal{H}^{(e)} \rightarrow \mathcal{H}^{(e+1)}$",
                "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=9.2)
    rounded_box(ax, 5.85, 1.25, 6.55, 0.62,
                r"Outputs of Phase 2: Learned Hints $\mathcal{H}^{*}$  |  $\mathcal{M}_{\phi}(\cdot;\mathcal{H}^{*})$",
                "", fc="#FFFFFF", ec=PURPLE, title_size=11)

    # Phase 2 execution arrows.
    arrow(ax, 6.58, 7.35, 6.58, 7.00)
    arrow(ax, 8.07, 7.35, 8.07, 7.00)
    arrow(ax, 9.58, 7.35, 9.58, 6.26, rad=0.0)
    arrow(ax, 11.08, 7.35, 10.30, 6.00, rad=-0.2, label=r"condition on $\mathcal{H}^{(e)}$",
          text_offset=(0.34, 0.0), lw=1.4)
    arrow(ax, 11.08, 7.35, 11.08, 5.24, color=BLUE, lw=1.2)
    arrow(ax, 8.18, 6.42, 8.18, 6.26)
    arrow(ax, 8.18, 5.68, 8.18, 5.52)
    arrow(ax, 8.18, 4.94, 8.18, 4.64)
    arrow(ax, 8.18, 3.86, 8.18, 3.76)
    arrow(ax, 8.18, 3.42, 8.18, 3.32)
    arrow(ax, 9.6, 3.84, 10.55, 3.84, label=r"self-play batch $\mathcal{B}^{(e)}$", text_offset=(0.0, 0.2))
    arrow(ax, 11.46, 4.20, 11.46, 3.97, color=RED)
    arrow(ax, 11.46, 3.45, 11.46, 3.20, color=RED)
    ax.plot([11.46, 12.35], [2.96, 2.96], color=RED, linewidth=1.5, zorder=7)
    ax.plot([12.35, 12.35], [2.96, 8.0], color=RED, linewidth=1.5, zorder=7)
    arrow(ax, 12.35, 8.0, 11.70, 8.0, color=RED, lw=1.5, ms=10,
          label="next epoch", text_offset=(0.0, 0.22))
    arrow(ax, 11.46, 2.72, 11.46, 1.87, color=RED)

    # Loading from Phase 1 into Phase 2, kept local to avoid clutter.
    rounded_box(ax, 5.74, 4.12, 0.62, 1.08, r"Load" + "\n" + r"$\theta^*,\mathcal{Z}$",
                "", fc="#F6FFF6", ec=GREEN, title_size=8.2)
    arrow(ax, 5.1, 1.45, 5.74, 4.66, color=GREEN, dashed=True, lw=1.4,
          label="from\nPhase 1", text_offset=(-0.25, 0.55), ms=9)
    arrow(ax, 6.36, 4.80, 6.05, 5.18, color=GREEN, dashed=True, lw=1.2, ms=9)
    arrow(ax, 6.36, 4.35, 5.95, 3.92, color=GREEN, dashed=True, lw=1.2, ms=9)

    # Inference.
    ix = 13.2
    iy = 7.35
    iw = 1.0
    inf_inputs = [
        ("Macro Goal", r"$g$"),
        ("Dialogue\nHistory", r"$h_t$"),
        ("Constraints", r"$c_t$"),
        ("Learned\nHints", r"$\mathcal{H}^{*}$ fixed"),
    ]
    for i, (t, s) in enumerate(inf_inputs):
        rounded_box(ax, ix + i * 1.12, iy, iw, 0.75, t, s, fc="#FFFFFF", ec=ORANGE, title_size=7.5, sub_size=7.3)

    rounded_box(ax, 13.65, 6.75, 3.4, 0.55, r"1. Intent-Drift Detector $\mathcal{D}_{\psi}$",
                r"consume $h_t$ and infer $(\hat{i}_t,\delta_t)$", fc="#FFFFFF", ec=BLUE, dashed=False,
                title_size=8.4, sub_size=7.0)
    rounded_box(ax, 13.65, 6.05, 3.4, 0.55, r"2. Hint-Conditioned Meta-Controller $\mathcal{M}_{\phi}(\cdot;\mathcal{H}^{*})$",
                "conditioned on fixed learned hints", fc="#FFFFFF", ec=BLUE, dashed=False,
                title_size=7.8, sub_size=7.0)
    rounded_box(ax, 13.65, 5.35, 3.4, 0.55, r"3. Skill / Weight Selection $z_t / w_t$",
                r"select over $\mathcal{Z}$", fc="#FFFFFF", ec=BLUE, dashed=False,
                title_size=8.4, sub_size=7.0)
    rounded_box(ax, 13.65, 4.65, 3.4, 0.55, r"4. Frozen Low-Level Policy $\theta^*$",
                "", fc="#FFFFFF", ec=BLUE, dashed=False, title_size=8.8)
    rounded_box(ax, 13.65, 3.95, 3.4, 0.55, r"5. Safety-Aware Dialogue Action $a_t$",
                "dialogue action + constraint checking", fc="#FFFFFF", ec=BLUE, dashed=False,
                title_size=8.2, sub_size=6.8)
    rounded_box(ax, 13.65, 3.15, 3.4, 0.62, "6. Opponent Response & State Update",
                "user response, updated dialogue state", fc="#FFFFFF", ec=BLUE, dashed=False,
                title_size=8.0, sub_size=6.8)
    rounded_box(ax, 13.35, 1.55, 3.95, 0.78, "Evaluation only",
                "GSR     /     T2DA     /     CVR\nno feedback, no updates", fc="#FFFFFF", ec=ORANGE,
                title_size=10, sub_size=7.8)

    # Inference blue arrows.
    for i in range(4):
        arrow(ax, ix + i * 1.12 + 0.5, iy, ix + i * 1.12 + 0.5, 7.3, color=BLUE, lw=1.2, ms=9)
    arrow(ax, 15.35, 6.75, 15.35, 6.60)
    arrow(ax, 15.35, 6.05, 15.35, 5.90)
    arrow(ax, 15.35, 5.35, 15.35, 5.20)
    arrow(ax, 15.35, 4.65, 15.35, 4.50)
    arrow(ax, 15.35, 3.95, 15.35, 3.77)
    arrow(ax, 15.35, 3.15, 15.35, 2.33, label="full dialogue\ntrajectory", text_offset=(0.62, 0.18))

    # Next-turn loop to dialogue history only.
    ax.plot([17.05, 17.45], [3.46, 3.46], color=BLUE, linewidth=1.5, zorder=6)
    ax.plot([17.45, 17.45], [3.46, 7.98], color=BLUE, linewidth=1.5, zorder=6)
    arrow(ax, 17.45, 7.98, 14.82, 7.98, color=BLUE, lw=1.5, ms=10, label="next turn context", text_offset=(0.6, 0.17))

    # Loading learned components into inference, routed only to learned modules.
    rounded_box(ax, 12.98, 4.55, 0.72, 1.35, r"Load" + "\n" + r"$\theta^*,\mathcal{Z}$" + "\n" + r"$\mathcal{H}^*,\mathcal{M}_{\phi}$",
                "", fc="#F6FFF6", ec=GREEN, title_size=8.3)
    arrow(ax, 13.70, 5.55, 13.65, 6.32, color=GREEN, dashed=True, lw=1.3, ms=9)  # H*, Mphi to controller
    arrow(ax, 13.70, 5.15, 13.65, 5.62, color=GREEN, dashed=True, lw=1.3, ms=9)  # Z to selection
    arrow(ax, 13.70, 4.83, 13.65, 4.92, color=GREEN, dashed=True, lw=1.3, ms=9)  # theta to policy

    # Legend.
    rounded_box(ax, 2.0, 0.18, 13.9, 0.55, "", "", fc="#FFFFFF", ec="#9CA3AF", dashed=False, lw=1.0)
    ax.text(2.35, 0.46, "Legend:", fontsize=10, fontweight="bold", ha="left", va="center")
    arrow(ax, 3.45, 0.46, 4.25, 0.46, color=BLUE, lw=1.8)
    ax.text(4.45, 0.46, "Execution / Data-Control Flow", fontsize=8.3, color=BLUE, va="center")
    arrow(ax, 6.25, 0.46, 7.05, 0.46, color=RED, lw=1.8)
    ax.text(7.25, 0.46, "Learning / Update during training only", fontsize=8.3, color=RED, va="center")
    arrow(ax, 10.0, 0.46, 10.80, 0.46, color=GREEN, dashed=True, lw=1.8)
    ax.text(11.0, 0.46, "Loading / Reuse of learned components", fontsize=8.3, color=GREEN, va="center")
    ax.text(14.0, 0.46, "*", fontsize=18, color=BLUE, fontweight="bold", va="center", ha="center")
    ax.text(14.25, 0.46, "Frozen component", fontsize=8.3, color=TEXT, va="center")

    fig.tight_layout(pad=0.0)
    for ext in ("jpg", "png", "pdf", "svg"):
        out = OUT_DIR / f"almond_correct_flow.{ext}"
        if ext == "jpg":
            fig.savefig(out, dpi=300, pil_kwargs={"quality": 95, "subsampling": 0})
        else:
            fig.savefig(out, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
