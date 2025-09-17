"""Loman: A Python library for building computation graphs.

Loman provides tools for creating and managing dependency-aware computation graphs
where nodes represent data or calculations, and edges represent dependencies.
"""

import loman.util as util
import loman.visualization as viz
from loman.computeengine import C, Computation, computation_factory, block, calc_node, input_node, node

# Backward compatibility alias
ComputationFactory = computation_factory
from loman.consts import NodeTransformations, States
from loman.exception import (
    CannotInsertToPlaceholderNodeException,
    LoopDetectedException,
    MapException,
    NonExistentNodeException,
)
from loman.nodekey import Name, Names, NodeKey, to_nodekey
from loman.visualization import GraphView

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
    "CannotInsertToPlaceholderNodeException",
    "LoopDetectedException",
    "MapException",
    "NonExistentNodeException",
    "Name",
    "Names",
    "NodeKey",
    "to_nodekey",
    "GraphView",
]
