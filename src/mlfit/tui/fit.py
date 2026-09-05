from enum import Enum


class FitLevel(str, Enum):
    """
    How well a model fits in the available VRAM, mirroring llmfit's four-level
    scale plus an UNKNOWN state for models with no size data.

    Each level maps to a colored dot rendered in the model browser. Colors use
    Textual theme variables so they adapt to the active theme.
    """

    PERFECT = "perfect"
    GOOD = "good"
    MARGINAL = "marginal"
    TIGHT = "tight"
    UNKNOWN = "unknown"

    @property
    def dot_markup(self) -> str:
        """Return the content-markup dot for this fit level (theme-aware)."""
        return _DOT_MARKUP[self]


_DOT_MARKUP: dict[FitLevel, str] = {
    FitLevel.PERFECT: "[$success]●[/]",
    FitLevel.GOOD: "[$warning]●[/]",
    FitLevel.MARGINAL: "[$accent]●[/]",
    FitLevel.TIGHT: "[$error]●[/]",
    FitLevel.UNKNOWN: "[$text-muted]○[/]",
}


def classify_fit(estimated_gb: float, vram_gb: float) -> FitLevel:
    """
    Classify how comfortably a model fits in available VRAM.

    Args:
        estimated_gb: Estimated memory the model needs, in GB.
        vram_gb: Available VRAM (or unified memory), in GB.

    Returns:
        A FitLevel based on the used-memory ratio:
          <= 60%  → PERFECT
          <= 85%  → GOOD
          <= 100% → MARGINAL
          >  100% → TIGHT
        UNKNOWN if VRAM is non-positive.
    """
    if vram_gb <= 0:
        return FitLevel.UNKNOWN

    ratio = estimated_gb / vram_gb
    if ratio <= 0.60:
        return FitLevel.PERFECT
    if ratio <= 0.85:
        return FitLevel.GOOD
    if ratio <= 1.0:
        return FitLevel.MARGINAL
    return FitLevel.TIGHT
