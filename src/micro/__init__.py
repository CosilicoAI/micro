"""
micro: Conditional microdata synthesis using normalizing flows.

A library for synthesizing survey microdata while preserving:
- Conditional relationships (demographics → outcomes)
- Zero-inflated distributions (common in economic/health data)
- Joint correlations between variables
- Hierarchical structures (households, firms, etc.)

Example:
    >>> from micro import Synthesizer
    >>> synth = Synthesizer(
    ...     target_vars=["income", "expenditure"],
    ...     condition_vars=["age", "education"],
    ... )
    >>> synth.fit(training_data)
    >>> synthetic = synth.generate(new_demographics)
"""

from micro.synthesizer import Synthesizer
from micro.transforms import (
    ZeroInflatedTransform,
    LogTransform,
    Standardizer,
    VariableTransformer,
    MultiVariableTransformer,
)
from micro.flows import ConditionalMAF, MADE, AffineCouplingLayer
from micro.discrete import (
    BinaryModel,
    CategoricalModel,
    DiscreteModelCollection,
)

__version__ = "0.1.0"

__all__ = [
    # Main class
    "Synthesizer",
    # Transforms
    "ZeroInflatedTransform",
    "LogTransform",
    "Standardizer",
    "VariableTransformer",
    "MultiVariableTransformer",
    # Flows
    "ConditionalMAF",
    "MADE",
    "AffineCouplingLayer",
    # Discrete
    "BinaryModel",
    "CategoricalModel",
    "DiscreteModelCollection",
]
