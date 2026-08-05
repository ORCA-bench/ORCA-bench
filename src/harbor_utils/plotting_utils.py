import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        # Fira Code (monospace, programming-friendly) across all plots that
        # import anything from this module. Falls back to the default
        # monospace family if Fira Code isn't installed.
        "font.family": ["Fira Code", "monospace"],
    }
)

# Single column figures
TICK_FONT_SIZE = 7
MINOR_TICK_FONT_SIZE = 3
LABEL_FONT_SIZE = 7.75
LEGEND_FONT_SIZE = 7
MARKER_ALPHA = 1
MARKER_SIZE = 3
LINE_ALPHA = 0.75
OUTWARD = 4

# ICLR text-width sizing. Use ``ICLR_WIDTH`` (full text width, in inches)
# for full-column figures and ``ICLR_WIDTH / 2`` for half-width figures.
ICLR_WIDTH = 5.5
ICLR_HEIGHT = 2

# Shared palette for plots that color by agent-model and shape by reasoning
# effort. Use ``MODEL_COLORS[i % len(MODEL_COLORS)]`` to assign per-model
# colors and ``EFFORT_MARKERS.get(effort, "o")`` for the marker style.
MODEL_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]
EFFORT_MARKERS = {"high": "o", "medium": "s", "low": "^"}
EFFORT_ORDER = {"high": 0, "medium": 1, "low": 2}


def get_display_label(agent: str, model: str) -> str:
    """Get a consistent display label for an agent+model combination.

    Formats labels as ``"{agent} ({model display name})"``, where the model
    display name is produced by :func:`get_model_display_name`.

    Args:
        agent: The agent name (e.g., "zeroshot", "codex", "terminus-2").
        model: The model name (e.g., "openai/gpt-5.2-2025-08-07").

    Returns:
        A formatted display label string.

    """
    return f"{agent} ({get_model_display_name(model)})"


def get_model_display_name(model: str) -> str:
    """Get a display name for a model.

    Strips the provider prefix (anything before ``/``), known vendor prefixes
    (e.g. ``anthropic-``, ``openai-``), trailing date suffixes
    (``-YYYY-MM-DD`` or ``-YYYYMMDD``), and version tags
    (``-preview``, ``-latest``, ``-beta``, ``-exp``).

    Examples:
        - ``"gradient_ai/anthropic-claude-4.6-sonnet"`` -> ``"claude-4.6-sonnet"``
        - ``"openai/gpt-5.2-2025-08-07"`` -> ``"gpt-5.2"``
        - ``"gemini/gemini-3-pro-preview"`` -> ``"gemini-3-pro"``

    Args:
        model: The fully-qualified model name.

    Returns:
        A formatted display name string.

    """
    import re

    model_short = model.split("/")[-1]

    # Strip known vendor prefixes (e.g., "anthropic-claude-4.6-sonnet" -> "claude-4.6-sonnet")
    for prefix in ("anthropic-", "openai-", "google-", "meta-", "mistral-"):
        if model_short.startswith(prefix):
            model_short = model_short[len(prefix) :]
            break

    # Strip trailing date suffixes like "-2025-08-07" or "-20250807"
    model_short = re.sub(r"-\d{4}-\d{2}-\d{2}$", "", model_short)
    model_short = re.sub(r"-\d{8}$", "", model_short)

    # Strip common preview/version tags
    model_short = re.sub(r"-(preview|latest|beta|exp)$", "", model_short)

    return model_short
