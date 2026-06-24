r"""Plot per-agent Markov-chain command-transition diagrams from Harbor runs.

Each shell command is classified into one of four *main* categories (a coarsening
of ``classify_cmd``): ``Discovery``, ``Telemetry``, ``Code``, ``Other`` (shown in
the legend as "Telemetry discovery", "Telemetry query", "Source code", "Other").
Setup, utility, and control-flow commands all fold into the catch-all ``Other``.
Nodes are these categories (area proportional to the marginal share of all shell
commands); directed edges are transition probabilities ``P(next | prev)`` between
the command category in step *i* and the command category in step *i+1*. Edges
whose transition probability ``P(next | prev)`` is below ``--min-prob`` (default
10%) are dropped and logged; ``--min-flow`` is an optional secondary filter on an
edge's joint flow (``marginal_prev * P``), off by default. Joint flow still scales
edge widths regardless.

Transition weighting: a single step can issue multiple shell commands. For two
consecutive shell-containing steps with ``n_i`` and ``n_j`` commands, every one of
the ``n_i * n_j`` ordered (category_a, category_b) pairs contributes weight
``1 / (n_i * n_j)`` to the transition count matrix. Counts are aggregated over all
trials in an agent-model group and row-normalized to transition probabilities.

Usage:
    cd examples/otel-demo
    uv run python format_markov_chain.py -jd jobs -od out
    uv run python format_markov_chain.py -jd jobs-0218 -od out-0218 --min-prob 0.05

Examples::

    # One panel per agent-model group, restricted to trials in
    # out-0531/all-predictions.json, written to out-0531/figures/markov_chain.pdf
    uv run python format_markov_chain.py -jd jobs -od out-0531

    # Highlight two models in the main-paper figure and put the rest in a
    # companion appendix figure. --models filters by substring of the group label;
    # the joint-flow edge-width scale stays global so the two figures are
    # comparable.
    uv run python format_markov_chain.py -jd jobs -od out-0531 \\
        --models opus-4.7 glm-5 --fig-name markov_chain.pdf
    uv run python format_markov_chain.py -jd jobs -od out-0531 \\
        --models 4.6-sonnet gpt-5.5 deepseek-v4-pro --fig-name markov_chain_appendix.pdf
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Patch
from tabulate import tabulate

from harbor_utils.plotting_utils import (
    ICLR_HEIGHT,
    ICLR_WIDTH,
    LABEL_FONT_SIZE,
    LEGEND_FONT_SIZE,
    TICK_FONT_SIZE,
)
from utils import (
    CODE_CATEGORIES,
    TELEM_DATA_CATEGORIES,
    TELEM_DISCOVERY_CATEGORIES,
    get_base_parser,
    load_invalid_trial_ids,
    model_alias,
    model_sort_key,
)

from format_tool_calls import SHELL_TOOL_NAMES, collect_tool_calls

# ── Main categories ────────────────────────────────────────────────────
# Display / layout / legend order of the four nodes. Setup/Utility/control
# commands all fold into the catch-all "Other". compute_layout places index 0 at
# the 1:30 position and proceeds clockwise, so this order puts Discovery@1:30,
# Telemetry@4:30, Code@7:30, Other@10:30.
MAIN_CATEGORIES = ["Discovery", "Telemetry", "Code", "Other"]

# A command can carry fine-grained labels from several main categories; the
# first match in this order wins when collapsing to a single node.
MAIN_PRIORITY = ["Telemetry", "Code", "Discovery", "Other"]

# Warm orange for Code, blue for the Telemetry-query node, green for the
# Telemetry-discovery node (a distinct hue), grey for the catch-all Other. The
# two telemetry nodes are value-shaded along their respective ramps (see below);
# these entries are the representative mid-tone swatches used in the legend.
MAIN_CATEGORY_COLORS = {
    "Code": "#e6550d",
    "Discovery": "#74c476",
    "Telemetry": "#2b8cbe",
    "Other": "#969696",
}

# Human-readable legend labels for the internal category keys above.
CATEGORY_DISPLAY_NAMES = {
    "Code": "Source code",
    "Discovery": "Telemetry discovery",
    "Telemetry": "Telemetry query",
    "Other": "Other",
}

# The telemetry-discovery, telemetry-query, and source-code nodes are shaded by
# their marginal share on a per-figure scale (min/max over the plotted models),
# in distinct hues (green / blue / orange). Direction marks the preferable
# behavior: Discovery is darker for a *smaller* share (less discovery is better),
# while Telemetry query and Source code are darker for a *larger* share (more is
# better). SHADE_RAMP gives the lo = light, hi = dark colormap positions.
DISCOVERY_CMAP = "Greens"
TELEMETRY_CMAP = "Blues"
CODE_CMAP = "Oranges"
SHADE_RAMP = (0.35, 0.90)

# Primary drop criterion: edges whose transition probability P(next | prev) is
# below this are dropped from the diagram and logged.
MIN_TRANS_PROB = 0.10
# Optional secondary filter on an edge's joint flow (marginal_prev * P), off by
# default. Joint flow still drives edge widths regardless of this threshold.
MIN_JOINT_FLOW = 0.0

# Radius of the ring the node centers sit on. Kept small enough that a node
# (radius up to R_MAX) plus its outward self-loop bulge
# (|SELF_LOOP_RAD| * SELF_LOOP_SEP) fits inside the tight VIEW_LIM box. On the
# diagonal (1:30/4:30/...) layout that worst-case extent projects onto each axis
# scaled by cos(45°): (LAYOUT_RADIUS + R_MAX + |SELF_LOOP_RAD| * SELF_LOOP_SEP)
# * cos(45°) <= VIEW_LIM, i.e. 1.32 * 0.707 ≈ 0.93 <= VIEW_LIM.
LAYOUT_RADIUS = 0.8
# Node sizing: max circle radius (data units) for the most-frequent category.
R_MAX = 0.32
# Half-width of the (square) per-panel view; trims empty margin around the graph.
VIEW_LIM = 1.02
# arc3 curvature of inter-node edges; bows a->b and b->a to opposite sides.
EDGE_RAD = 0.18
# Self-loop geometry: negative arc3 curvature balloons the loop *outward* (away
# from the diagram center); SELF_LOOP_SEP is the half-separation of its two
# rim anchor points.
SELF_LOOP_RAD = -2.0
SELF_LOOP_SEP = 0.1
# Edge line-width range (points).
LW_MIN, LW_MAX = 0.4, 4.0
# Self-loops sit on a small arc, so a full-width stem fills the loop and hides
# the arrowhead; cap the stem narrower than inter-node edges.
SELF_LOOP_LW_MAX = 1.6


def collapse_to_main(categories: list[str]) -> str:
    """Collapse a shell command's fine-grained labels to one main category.

    Args:
        categories: fine-grained labels from ``classify_cmd`` (may be empty or
            carry labels from multiple main categories).

    Returns:
        The highest-priority main category present (``MAIN_PRIORITY`` order);
        ``"Other"`` when no labels match. Control-flow labels (``control:*``,
        e.g. waits/interrupts) are folded into ``"Other"`` as harness mechanics.

    """
    mains: set[str] = set()
    for c in categories:
        if c in CODE_CATEGORIES:
            mains.add("Code")
        elif c in TELEM_DATA_CATEGORIES:
            mains.add("Telemetry")
        elif c in TELEM_DISCOVERY_CATEGORIES:
            mains.add("Discovery")
        else:
            # Setup / Utility / control mechanics (waits/interrupts) and any
            # unmatched label all fold into the catch-all "Other".
            mains.add("Other")
    if not mains:
        return "Other"
    return next(m for m in MAIN_PRIORITY if m in mains)


def group_title(group: str) -> str:
    """Compact panel title for a group label, via ``utils.model_alias``.

    The group label is ``"{agent} ({model}) [(effort)]"``; alias the model and
    drop the agent and any effort suffix (e.g.
    ``"terminus-2 (claude-opus-4.7) (medium)"`` -> ``"Opus 4.7"``).
    """
    parens = re.findall(r"\(([^)]*)\)", group)
    if not parens:
        return group
    return model_alias(parens[0])


def build_step_commands(
    rows: list[dict],
) -> dict[str, dict[str, dict[int, list[str]]]]:
    """Group shell commands into ``group -> trial_id -> step_id -> [main_cat]``.

    Non-shell tool calls (``categories is None``) are ignored.
    """
    out: dict[str, dict[str, dict[int, list[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for row in rows:
        if row["function_name"] not in SHELL_TOOL_NAMES:
            continue
        main = collapse_to_main(row["categories"] or [])
        out[row["group"]][row["trial_id"]][row["step_id"]].append(main)
    return out


def build_transition_matrix(
    trials: dict[str, dict[int, list[str]]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the transition count matrix, transition probabilities and marginal.

    Args:
        trials: ``trial_id -> step_id -> [main_cat, ...]`` for one group.

    Returns:
        ``(C, P, marginal)`` as ``4x4``, ``4x4`` and ``4`` numpy arrays indexed
        by ``MAIN_CATEGORIES``. ``C[a][b]`` is the ``1/(n_i*n_j)``-weighted count
        of category-``a``→category-``b`` transitions across consecutive
        shell-containing steps; ``P`` is the row-normalized ``C``; ``marginal``
        is the fraction of all shell commands in each main category.

    """
    idx = {cat: k for k, cat in enumerate(MAIN_CATEGORIES)}
    n = len(MAIN_CATEGORIES)
    counts = np.zeros((n, n))
    node_counts = np.zeros(n)

    for steps_map in trials.values():
        ordered = [steps_map[s] for s in sorted(steps_map)]
        for cmds in ordered:
            for cat in cmds:
                node_counts[idx[cat]] += 1
        for prev_cmds, next_cmds in zip(ordered, ordered[1:]):
            w = 1.0 / (len(prev_cmds) * len(next_cmds))
            for a, ca in Counter(prev_cmds).items():
                for b, cb in Counter(next_cmds).items():
                    counts[idx[a]][idx[b]] += w * ca * cb

    total = node_counts.sum()
    marginal = node_counts / total if total else node_counts
    row_sums = counts.sum(axis=1, keepdims=True)
    probs = np.divide(counts, row_sums, out=np.zeros_like(counts), where=row_sums > 0)
    return counts, probs, marginal


def compute_layout() -> np.ndarray:
    """Return node centers on the unit circle, clockwise from the 1:30 position.

    The 45° offset from the top places the four nodes at the diagonal clock
    positions (1:30, 4:30, 7:30, 10:30) rather than 12/3/6/9, shrinking the
    horizontal extent to ``2 * LAYOUT_RADIUS * cos(45°)``.
    """
    angles = np.pi / 4 - np.arange(len(MAIN_CATEGORIES)) * (
        2 * np.pi / len(MAIN_CATEGORIES)
    )
    return LAYOUT_RADIUS * np.column_stack([np.cos(angles), np.sin(angles)])


def _lw(joint: float, jmax: float) -> float:
    """Map a joint-flow value to an edge line width."""
    return LW_MIN + (LW_MAX - LW_MIN) * (joint / jmax if jmax else 0.0)


def _contrast_color(facecolor: str) -> str:
    """Return white or dark text, whichever contrasts the given fill color."""
    r, g, b = to_rgb(facecolor)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    # 0.62 threshold keeps the medium-gray Other node (#969696, lum ~0.59) on
    # white text; lighter fills (lum > ~0.62) keep dark text.
    return "white" if luminance < 0.62 else "#333333"


def _value_shade(
    share: float, vmin: float, vmax: float, *, cmap_name: str, darker_is_low: bool
) -> tuple:
    """Sequential shade for a node's marginal share on a per-figure ``[vmin, vmax]`` scale.

    ``cmap_name`` selects the colormap (e.g. greens for discovery, blues for
    query). ``darker_is_low=True`` makes a *smaller* share darker (Discovery:
    less is better); ``darker_is_low=False`` makes a *larger* share darker
    (Telemetry query: more is better). Equal bounds (single model) map to the
    mid-tone.
    """
    t = (share - vmin) / (vmax - vmin) if vmax > vmin else 0.5
    lo, hi = SHADE_RAMP
    pos = hi - (hi - lo) * t if darker_is_low else lo + (hi - lo) * t
    return plt.get_cmap(cmap_name)(pos)


def draw_markov_chain(
    ax: plt.Axes,
    probs: np.ndarray,
    marginal: np.ndarray,
    pos: np.ndarray,
    *,
    jmax: float,
    min_prob: float,
    min_flow: float,
    node_colors: dict[str, object],
) -> list[tuple[str, str, float, float]]:
    """Draw one Markov-chain panel; return dropped ``(src, dst, prob, joint)`` edges.

    ``node_colors`` maps each main category to its fill color for this panel
    (e.g. the Discovery entry is value-shaded by :func:`_discovery_color`); it is
    used for node fills, label contrast, and outgoing edge/self-loop colors.
    """
    ax.set_aspect("equal")
    ax.set_xlim(-VIEW_LIM, VIEW_LIM)
    ax.set_ylim(-VIEW_LIM, VIEW_LIM)
    ax.axis("off")
    ax.margins(0)

    share_max = marginal.max() or 1.0
    radii = R_MAX * np.sqrt(marginal / share_max)

    # Nodes + labels.
    for k, cat in enumerate(MAIN_CATEGORIES):
        if radii[k] <= 0:
            continue
        ax.add_patch(
            Circle(
                pos[k],
                radii[k],
                facecolor=node_colors[cat],
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
        )
        # Percentage label = marginal share. Centered in large nodes (text color
        # contrasts the fill); nudged just outside nodes too small to hold the
        # text, where it sits on the white background and stays dark.
        pct = f"{100 * marginal[k]:.0f}%"
        if radii[k] < 0.08:
            radial = pos[k] / np.linalg.norm(pos[k])
            label_xy = pos[k] + radial * (radii[k] + 0.08)
            text_color = "#333333"
        else:
            label_xy = pos[k]
            text_color = _contrast_color(node_colors[cat])
        ax.text(
            *label_xy,
            pct,
            ha="center",
            va="center",
            fontsize=TICK_FONT_SIZE,
            color=text_color,
            zorder=4,
        )

    dropped: list[tuple[str, str, float]] = []
    n = len(MAIN_CATEGORIES)

    # Directed edges between distinct nodes.
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            joint = marginal[a] * probs[a, b]
            if probs[a, b] < min_prob or joint < min_flow:
                if probs[a, b] > 0:
                    dropped.append(
                        (MAIN_CATEGORIES[a], MAIN_CATEGORIES[b], probs[a, b], joint)
                    )
                continue
            d = pos[b] - pos[a]
            dist = np.linalg.norm(d)
            u = d / dist
            start = pos[a] + u * radii[a]
            end = pos[b] - u * radii[b]
            # Same-sign curvature for both directions: reversing start/end
            # already flips the arc's side, so a->b and b->a bow to opposite
            # physical sides (avoiding a shared overlapping stem).
            rad = EDGE_RAD
            ax.add_patch(
                FancyArrowPatch(
                    start,
                    end,
                    connectionstyle=f"arc3,rad={rad}",
                    arrowstyle="-|>",
                    mutation_scale=8 + 6 * (joint / jmax if jmax else 0.0),
                    linewidth=_lw(joint, jmax),
                    color=node_colors[MAIN_CATEGORIES[a]],
                    alpha=0.7,
                    zorder=2,
                )
            )
            # Transition-probability label at the arc's bowed midpoint. matplotlib's
            # arc3 bows to the right of start->end (control point at mid+rad*(dy,-dx)),
            # so offset the label along that same (dy, -dx) perpendicular — otherwise
            # the label lands beside the opposite-direction arc.
            mid = (start + end) / 2
            perp = np.array([(end - start)[1], -(end - start)[0]])
            perp = perp / np.linalg.norm(perp)
            label_pos = mid + perp * rad * dist * 0.5
            ax.text(
                *label_pos,
                f"{100 * probs[a, b]:.0f}%",
                ha="center",
                va="center",
                fontsize=TICK_FONT_SIZE,
                color="#333333",
                zorder=5,
            )

    # Self-loops (diagonal).
    for a in range(n):
        joint = marginal[a] * probs[a, a]
        if probs[a, a] < min_prob or joint < min_flow or radii[a] <= 0:
            if probs[a, a] > 0 and (probs[a, a] < min_prob or joint < min_flow):
                dropped.append(
                    (MAIN_CATEGORIES[a], MAIN_CATEGORIES[a], probs[a, a], joint)
                )
            continue
        radial = pos[a] / np.linalg.norm(pos[a])
        tangent = np.array([-radial[1], radial[0]])
        base = pos[a] + radial * radii[a]
        p1 = base + tangent * SELF_LOOP_SEP
        p2 = base - tangent * SELF_LOOP_SEP
        ax.add_patch(
            FancyArrowPatch(
                p1,
                p2,
                connectionstyle=f"arc3,rad={SELF_LOOP_RAD}",
                arrowstyle="-|>",
                # Scale the arrowhead with flow (as inter-node edges do) and cap
                # the stem so the head stays clearly larger than the line.
                mutation_scale=8 + 6 * (joint / jmax if jmax else 0.0),
                linewidth=min(_lw(joint, jmax), SELF_LOOP_LW_MAX),
                color=node_colors[MAIN_CATEGORIES[a]],
                alpha=0.7,
                # Default shrinkA/B (2pt) pulls the loop ends off the rim,
                # leaving a visible gap; anchor both directly on the node.
                shrinkA=0,
                shrinkB=0,
                zorder=2,
            )
        )
        ax.text(
            *(base + radial * 0.2),
            f"{100 * probs[a, a]:.0f}%",
            ha="center",
            va="center",
            fontsize=TICK_FONT_SIZE,
            color="#333333",
            zorder=5,
        )
    return dropped


def main() -> None:
    """Plot per-agent Markov-chain command-transition diagrams."""
    parser = get_base_parser()
    parser.description = "Plot per-agent Markov-chain command-transition diagrams."
    parser.add_argument(
        "--jobs-dir",
        "-jd",
        type=Path,
        default=Path("jobs"),
        help="Path to the Harbor jobs directory (default: jobs)",
    )
    parser.add_argument(
        "--min-prob",
        type=float,
        default=MIN_TRANS_PROB,
        help="Drop edges with transition probability P(next|prev) below this "
        "(default: %(default)s)",
    )
    parser.add_argument(
        "--min-flow",
        type=float,
        default=MIN_JOINT_FLOW,
        help="Optionally also drop edges with joint flow (marginal_prev * P) "
        "below this; off by default (default: %(default)s)",
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help=(
            "Only plot groups whose label contains one of these substrings "
            "(e.g. opus-4.7 deepseek-v4-pro). Default: all groups. The joint-flow "
            "edge-width scale is still computed over all groups so panels stay "
            "comparable across separate (e.g. main vs. appendix) figures."
        ),
    )
    parser.add_argument(
        "--fig-name",
        default="markov_chain.pdf",
        help="Output figure filename under OUTPUT_DIR/figures/ (default: %(default)s)",
    )
    args = parser.parse_args()

    invalid_ids: set[str] = (
        load_invalid_trial_ids(args.filter_xlsx) if args.filter_xlsx else set()
    )

    preds_path = args.output_dir / "all-predictions.json"
    with preds_path.open() as f:
        valid_trial_ids = set(json.load(f).keys())
    print(f"Restricting to {len(valid_trial_ids)} trial(s) in {preds_path}")

    rows = collect_tool_calls(
        args.jobs_dir, csv_dir=None, valid_trial_ids=valid_trial_ids
    )
    if invalid_ids:
        rows = [r for r in rows if r["trial_id"] not in invalid_ids]
    if not rows:
        print("No tool calls found.")
        return

    step_commands = build_step_commands(rows)

    # Per-group matrices; keep only groups that issued at least one shell command.
    mats: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for group, trials in step_commands.items():
        counts, probs, marginal = build_transition_matrix(trials)
        if marginal.sum() > 0:
            mats[group] = (counts, probs, marginal)
    groups = sorted(mats, key=model_sort_key)
    if not groups:
        print("No shell commands found in any group.")
        return

    # Global joint-flow scale so edge widths are comparable across panels (and
    # across separate figures rendered from the same data) — computed over all
    # groups before any --models filter is applied.
    jmax = max(float((m * p).max()) for _, p, m in mats.values()) or 1.0

    if args.models:
        groups = [g for g in groups if any(m.lower() in g.lower() for m in args.models)]
        if not groups:
            print(f"No groups match --models {args.models}.")
            return

    # Per-figure value-shading scales (normalized over the plotted models):
    # Discovery darker = smaller share; Telemetry query and Source code darker =
    # larger share.
    disc_idx = MAIN_CATEGORIES.index("Discovery")
    tele_idx = MAIN_CATEGORIES.index("Telemetry")
    code_idx = MAIN_CATEGORIES.index("Code")
    disc_shares = [mats[g][2][disc_idx] for g in groups]
    tele_shares = [mats[g][2][tele_idx] for g in groups]
    code_shares = [mats[g][2][code_idx] for g in groups]
    disc_min, disc_max = min(disc_shares), max(disc_shares)
    tele_min, tele_max = min(tele_shares), max(tele_shares)
    code_min, code_max = min(code_shares), max(code_shares)

    # Full text-width row, one panel per group.
    n_groups = len(groups)
    fig, axes = plt.subplots(
        1,
        n_groups,
        figsize=(ICLR_WIDTH, ICLR_HEIGHT),
        squeeze=False,
        gridspec_kw={"wspace": 0.0},
    )
    pos = compute_layout()

    for i, group in enumerate(groups):
        _, probs, marginal = mats[group]
        node_colors = {
            **MAIN_CATEGORY_COLORS,
            "Discovery": _value_shade(
                marginal[disc_idx],
                disc_min,
                disc_max,
                cmap_name=DISCOVERY_CMAP,
                darker_is_low=True,
            ),
            "Telemetry": _value_shade(
                marginal[tele_idx],
                tele_min,
                tele_max,
                cmap_name=TELEMETRY_CMAP,
                darker_is_low=False,
            ),
            "Code": _value_shade(
                marginal[code_idx],
                code_min,
                code_max,
                cmap_name=CODE_CMAP,
                darker_is_low=False,
            ),
        }
        dropped = draw_markov_chain(
            axes[0, i],
            probs,
            marginal,
            pos,
            jmax=jmax,
            min_prob=args.min_prob,
            min_flow=args.min_flow,
            node_colors=node_colors,
        )
        # Panel labels "(a) Model", "(b) Model", ... in subfigure order.
        panel_label = f"({chr(ord('a') + i)}) {group_title(group)}"
        axes[0, i].text(
            0.5,
            -0.02,
            panel_label,
            transform=axes[0, i].transAxes,
            ha="center",
            va="top",
            fontsize=LABEL_FONT_SIZE,
            fontweight="bold",
        )

        print(f"\n[{panel_label}] [{group}] node shares:")
        for k, cat in enumerate(MAIN_CATEGORIES):
            print(f"  {cat}: {100 * marginal[k]:.1f}%")
        print(f"[{group}] transition matrix P(next | prev), rows=prev (%):")
        table = [
            [
                MAIN_CATEGORIES[a],
                *(f"{100 * probs[a, b]:.1f}" for b in range(len(MAIN_CATEGORIES))),
            ]
            for a in range(len(MAIN_CATEGORIES))
        ]
        print(
            tabulate(
                table,
                headers=["prev \\ next", *MAIN_CATEGORIES],
                tablefmt="github",
            )
        )
        for src, dst, prob, jf in dropped:
            print(
                f"[{group}] dropped edge {src}->{dst} "
                f"(P={100 * prob:.1f}% < {100 * args.min_prob:.0f}%"
                f" or joint flow {jf:.4f} < {args.min_flow})"
            )

    handles = [
        Patch(
            facecolor=MAIN_CATEGORY_COLORS[c],
            edgecolor="white",
            label=CATEGORY_DISPLAY_NAMES[c],
        )
        for c in MAIN_CATEGORIES
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=len(MAIN_CATEGORIES),
        fontsize=LEGEND_FONT_SIZE,
        frameon=False,
        bbox_to_anchor=(0.5, 1.04),
    )
    # tight_layout would re-inflate the inter-panel gaps; set margins manually.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, wspace=0.0)

    fig_path = args.output_dir / "figures" / args.fig_name
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    print(f"\nSaved figure to {fig_path}")
    plt.show()


if __name__ == "__main__":
    main()
