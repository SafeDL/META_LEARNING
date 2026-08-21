"""Compatibility bridge for audited legacy simulator primitives.

The active package consumes these immutable geometry/critical definitions
without modifying ``pearl_learning``.  The bridge can be removed only after
new golden-event tests supersede every imported primitive.
"""
from pearl_learning.src.critical import critical_measurements, strict_near_miss_potential
from pearl_learning.src.io import content_hash
from pearl_learning.src.routes import RoutePolyline

__all__ = ("RoutePolyline", "content_hash", "critical_measurements", "strict_near_miss_potential")
