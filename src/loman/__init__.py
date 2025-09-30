"""Loman: A Python library for building computation graphs.

Loman provides tools for creating and managing dependency-aware computation graphs
where nodes represent data or calculations, and edges represent dependencies.
"""

import loman.util as util
import loman.visualization as viz
from loman.computeengine import C, Computation, block, calc_node, computation_factory, input_node, node
from loman.consts import NodeTransformations, States
from loman.exception import (
    CannotInsertToPlaceholderNodeError,
    LoopDetectedError,
    MapError,
    NonExistentNodeError,
)
from loman.nodekey import Name, Names, NodeKey, to_nodekey
from loman.visualization import GraphView

# Backward compatibility alias
ComputationFactory = computation_factory

__all__ = [
    "util",
    "viz",
    "C",
    "Computation",
    "computation_factory",
    "ComputationFactory",  # Backward compatibility
    "block",
    "calc_node",
    "input_node",
    "node",
    "NodeTransformations",
    "States",
    "CannotInsertToPlaceholderNodeError",
    "LoopDetectedError",
    "MapError",
    "NonExistentNodeError",
    "Name",
    "Names",
    "NodeKey",
    "to_nodekey",
    "GraphView",
]
