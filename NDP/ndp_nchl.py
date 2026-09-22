"""Compatibility imports for experiment files saved before the R-NDP rename.

Keep this file in the project as ``NDP/ndp_nchl.py``. Historical pickle files
refer to the original module and class names and require both to remain
importable.
"""

from NDP.rewiring_ndp import RewiringNeuralDevelopmentalProgram


HebbianNeuralDevelopmentalProgram = RewiringNeuralDevelopmentalProgram


__all__ = [
    "RewiringNeuralDevelopmentalProgram",
    "HebbianNeuralDevelopmentalProgram",
]
