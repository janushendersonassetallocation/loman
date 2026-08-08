"""Copying a computation, and reading and writing it to a file.

Two formats are supported. JSON is the current one and delegates to
:class:`~loman.serialization.computation.ComputationSerializer`; the dill-based
methods are retained for backwards compatibility and are deprecated.
"""

import warnings
from collections import defaultdict
from typing import TYPE_CHECKING, Any, BinaryIO, Self, TextIO

import dill  # nosec B403
import networkx as nx

from .consts import NodeAttributes, SystemTags
from .exception import ValidationError
from .execution import ExecutionMixin

if TYPE_CHECKING:
    from .computeengine import Computation
    from .serialization.computation import ComputationSerializer


class PersistenceMixin(ExecutionMixin):
    """Copy, pickle and file round-trips for a computation."""

    def copy(self) -> Self:
        """Create a copy of a computation.

        The copy is shallow. Any values in the new Computation's DAG will be the same object as this Computation's
        DAG. As new objects will be created by any further computations, this should not be an issue.

        :rtype: Computation
        """
        obj = type(self)()
        obj.dag = nx.DiGraph(self.dag)
        obj._tag_map = defaultdict(set, {tag: nodes.copy() for tag, nodes in self._tag_map.items()})
        obj._state_map = {state: nodes.copy() for state, nodes in self._state_map.items()}
        return obj

    def __getstate__(self) -> dict[str, Any]:
        """Prepare computation for serialization by removing non-serializable nodes."""
        node_serialize = nx.get_node_attributes(self.dag, NodeAttributes.TAG)
        obj = self.copy()
        for name, tags in node_serialize.items():
            if SystemTags.SERIALIZE not in tags:
                obj._set_uninitialized(name)
        return {"dag": obj.dag}

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore computation from serialized state."""
        self.__init__()
        self.dag = state["dag"]
        self._refresh_maps()

    def write_dill_old(self, file_: str | BinaryIO) -> None:
        """Serialize a computation to a file or file-like object.

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        warnings.warn("write_dill_old is deprecated, use write_dill instead", DeprecationWarning, stacklevel=2)
        # Temporarily unhook __getstate__/__setstate__ so dill pickles the whole
        # object rather than the trimmed dict they produce. They are defined on
        # this mixin, not on Computation, so that is the class to detach them
        # from — deleting from self.__class__ would not find them.
        owner = PersistenceMixin
        original_getstate = owner.__getstate__
        original_setstate = owner.__setstate__

        try:
            del owner.__getstate__
            del owner.__setstate__

            node_serialize = nx.get_node_attributes(self.dag, NodeAttributes.TAG)
            obj = self.copy()
            obj.executor_map = None  # type: ignore[assignment]
            obj.default_executor = None  # type: ignore[assignment]
            for name, tags in node_serialize.items():
                if SystemTags.SERIALIZE not in tags:
                    obj._set_uninitialized(name)

            if isinstance(file_, str):
                with open(file_, "wb") as f:
                    dill.dump(obj, f)
            else:
                dill.dump(obj, file_)
        finally:
            owner.__getstate__ = original_getstate  # type: ignore[method-assign]
            owner.__setstate__ = original_setstate

    def write_dill(self, file_: str | BinaryIO) -> None:
        """Serialize a computation to a file or file-like object.

        .. deprecated::
            Use :meth:`write_json` instead.  dill-based serialization will be
            removed in a future release.

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        warnings.warn(
            "write_dill is deprecated and will be removed in a future release. Use write_json instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(file_, str):
            with open(file_, "wb") as f:
                dill.dump(self, f)
        else:
            dill.dump(self, file_)

    @classmethod
    def read_dill(cls, file_: str | BinaryIO) -> Self:
        """Deserialize a computation from a file or file-like object.

        .. deprecated::
            Use :meth:`read_json` instead.  dill-based serialization will be
            removed in a future release.

        .. warning::
            This method uses dill.load() which can execute arbitrary code.
            Only load files from trusted sources. Never load data from
            untrusted or unauthenticated sources as it may lead to arbitrary
            code execution.

        :param file_: If string, writes to a file
        :type file_: File-like object, or string
        """
        warnings.warn(
            "read_dill is deprecated and will be removed in a future release. Use read_json instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if isinstance(file_, str):
            with open(file_, "rb") as f:
                obj = dill.load(f)  # noqa: S301  # nosec B301
        else:
            obj = dill.load(file_)  # noqa: S301  # nosec B301
        if isinstance(obj, cls):
            return obj
        else:
            msg = "Loaded object is not a Computation"
            raise ValidationError(msg)

    def write_json(self, file_: str | TextIO, *, serializer: "ComputationSerializer | None" = None) -> None:
        """Serialize a computation to a JSON file or file-like object.

        Custom types can be supported by passing a custom *serializer* —
        either a :class:`~loman.serialization.computation.ComputationSerializer`
        instance with extra transformers registered, or a subclass that
        overrides the transformer factory.

        :param file_: Destination file path (str) or text-mode file-like object.
        :param serializer: Optional custom serializer.  If ``None`` the default
            :class:`~loman.serialization.computation.ComputationSerializer` is used.
        """
        from .serialization.computation import ComputationSerializer

        s = serializer if serializer is not None else ComputationSerializer()
        if isinstance(file_, str):
            with open(file_, "w", encoding="utf-8") as f:
                s.dump(self, f)
        else:
            s.dump(self, file_)

    @staticmethod
    def read_json(file_: str | TextIO, *, serializer: "ComputationSerializer | None" = None) -> "Computation":
        """Deserialize a computation from a JSON file or file-like object.

        :param file_: Source file path (str) or text-mode file-like object.
        :param serializer: Optional custom serializer.  If ``None`` the default
            :class:`~loman.serialization.computation.ComputationSerializer` is used.
        :rtype: Computation
        """
        from .serialization.computation import ComputationSerializer

        s = serializer if serializer is not None else ComputationSerializer()
        if isinstance(file_, str):
            with open(file_, encoding="utf-8") as f:
                return s.load(f)
        else:
            return s.load(file_)
