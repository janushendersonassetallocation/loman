"""Loman: A Python library for building computation graphs.

Loman provides tools for creating and managing dependency-aware computation graphs
where nodes represent data or calculations, and edges represent dependencies.
"""

import loman.util as util
import loman.visualization as viz
from loman.computeengine import (
    C,
    Computation,
    ComputationEvent,
    block,
    calc_node,
    computation_factory,
    input_node,
    node,
    repeated_blocks,
)
from loman.consts import NodeTransformations, States
from loman.exception import (
    CannotInsertToPlaceholderNodeError,
    DeserializedError,
    FittingError,
    InvalidBlockTypeError,
    LoopDetectedError,
    MapError,
    NonExistentNodeError,
    SerializationError,
    ValidationError,
)
from loman.nodekey import Name, Names, NodeKey, to_nodekey
from loman.planning import ExecutionPlan, ValidationReport
from loman.serialization import ComputationSerializer, SerializationProfile
from loman.util import BlockContext, BlockFeature, FanIn, FanOut, IdNode, InputValue, PlannedNode, Positional
from loman.visualization import GraphView

# Backward compatibility alias
ComputationFactory = computation_factory

__all__ = [
    "BlockContext",
    "BlockFeature",
    "C",
    "CannotInsertToPlaceholderNodeError",
    "Computation",
    "ComputationEvent",
    "ComputationFactory",  # Backward compatibility
    "ComputationSerializer",
    "DeserializedError",
    "ExecutionPlan",
    "FanIn",
    "FanOut",
    "FittingError",
    "GraphView",
    "IdNode",
    "InputValue",
    "InvalidBlockTypeError",
    "LoopDetectedError",
    "MapError",
    "Name",
    "Names",
    "NodeKey",
    "NodeTransformations",
    "NonExistentNodeError",
    "PlannedNode",
    "Positional",
    "SerializationError",
    "SerializationProfile",
    "States",
    "ValidationError",
    "ValidationReport",
    "block",
    "calc_node",
    "computation_factory",
    "input_node",
    "node",
    "repeated_blocks",
    "to_nodekey",
    "util",
    "viz",
]
