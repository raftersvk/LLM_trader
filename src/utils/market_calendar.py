"""Market calendar helpers.

Crypto markets trade 24/7, so the day of the week must not influence analysis,
prompts, memory retrieval or analytics. Upstream still models a weekend flag in
several structures; deleting those fields and parameters would conflict with
every future upstream sync, so the concept is neutralized here instead.
"""

from datetime import datetime


def is_weekend(_now: datetime | None = None) -> bool:
    """Return whether the market is currently in weekend mode.

    Always ``False``: cryptocurrency trades continuously, so a weekend flag is
    meaningless and must never shape a trading decision. This single helper is
    the deliberate divergence point from upstream; call sites use it instead of
    computing the weekday so upstream-owned fields and signatures stay intact.

    Args:
        _now: Optional reference time, accepted for call-site symmetry.
            Underscore-prefixed because it is intentionally unused: the result
            is constant.

    Returns:
        Always ``False``.
    """
    return False
