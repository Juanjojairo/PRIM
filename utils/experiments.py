"""Small helpers to build readable experiment names."""

from __future__ import annotations


def format_float_token(value: float) -> str:
    """Format a float into a compact filesystem-friendly token."""
    if float(value).is_integer():
        return str(int(value))
    scientific = f"{value:.0e}"
    if "e" in scientific:
        base, exponent = scientific.split("e")
        exponent = exponent.lstrip("+").replace("-0", "-").replace("+0", "")
        return f"{base}e{exponent}"
    return str(value).replace(".", "p")


def join_name_parts(*parts: object) -> str:
    """Join non-empty experiment name pieces with underscores."""
    normalized = []
    for part in parts:
        if part is None:
            continue
        text = str(part).strip()
        if text:
            normalized.append(text.replace("/", "-").replace(" ", "-"))
    if not normalized:
        raise ValueError("Expected at least one non-empty experiment name part.")
    return "_".join(normalized)
