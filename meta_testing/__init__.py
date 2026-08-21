"""Hierarchical, map-aware meta-testing for black-box driving controllers.

``pearl_learning`` remains an intentionally independent legacy baseline.  This
package owns the active MVR method and must never import a SUT identity into a
learned policy or posterior feature vector.
"""

from .scenario.task_spec import MetaTestTaskSpec
from .failure.signature import FailureSignature

__all__ = ("FailureSignature", "MetaTestTaskSpec")
