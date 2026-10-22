"""Historical-response transfer for fixed-budget collision discovery."""

from .posterior import CollisionPosterior
from .selection import select

__all__ = ["CollisionPosterior", "select"]
