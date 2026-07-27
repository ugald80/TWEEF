"""Backward-compatible import for the misspelled TrustworthyClassifier module.

Use ``TWEEF.TrustworthyClassifier`` instead. This module remains temporarily so
existing code that imports ``TWEEF.TrustworthyClassiffier`` keeps working.
"""

from TWEEF.TrustworthyClassifier import TrustworthyClassifier

__all__ = ["TrustworthyClassifier"]
