"""Turning a node definition typed in the browser into a real Loman node.

The widget's other requests name something that already exists --- a rendered
node, an open block, a cell of a value. These ones do not: a definition arrives
as text, and this module is what turns that text into the arguments
:meth:`~loman.computeengine.Computation.add_node` takes.

Two shapes of definition go over the wire:

``input``
    A name, and optionally a scalar to seed it with. Without one the node is
    created UNINITIALIZED, which is the ordinary way to declare an input you
    will supply later.
``calc``
    A name, a list of inputs and a Python expression. The inputs become the
    function's parameters and the graph's edges; the expression becomes its
    body.

Names are read relative to whatever the widget is rooted on, so a name typed
while focused on ``market`` lands inside ``market``. A leading ``/`` escapes
that and names a node from the top of the computation, which is how a block's
node depends on something outside it.

The compiled function is a real function with real source: the text is
registered with :mod:`linecache` under the node's own filename, so
:meth:`Computation.get_source` shows the user what they typed rather than
reporting that the source is unavailable. It is not, however, *importable* ---
a UI-built function is a lambda by another name, so a computation containing
one cannot round-trip its functions through :meth:`Computation.save`.
"""

from __future__ import annotations

import keyword
import linecache
import re
from dataclasses import dataclass, field
from textwrap import indent
from typing import TYPE_CHECKING, Any

from loman.consts import NodeAttributes
from loman.nodedefs import ConstantValue
from loman.nodekey import Name, NodeKey, to_nodekey

from .value import from_wire

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from loman.computeengine import Computation


class GraphBuildError(ValueError):
    """Raised when a node definition from the browser cannot be built."""


#: Longest expression the form accepts. Generous for the one-liners this is
#: for, and small enough that a runaway paste is refused rather than compiled.
MAX_EXPRESSION_LENGTH = 4_000

#: Most inputs one definition may declare. A node with more parameters than
#: this is not being written in a text box.
MAX_INPUTS = 64

#: Name given to the compiled function when the node's own label is not a
#: usable identifier --- ``market/1`` and ``portfolio value`` both land here.
_FALLBACK_FUNC_NAME = "node"

_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _is_identifier(text: str) -> bool:
    """Report whether ``text`` can be used as a Python parameter name."""
    return bool(_IDENTIFIER.match(text)) and not keyword.iskeyword(text)


def default_parameter(node_key: NodeKey) -> str:
    """Return the parameter name a node lends its own dependent, or ``""``.

    A node's last path part is what a function would naturally call it, and it
    is what Loman itself uses when resolving a parameter to a sibling node. A
    part that is not an identifier --- because it is numeric, quoted, or has a
    space in it --- lends nothing, and the definition has to name a parameter
    explicitly.

    :param node_key: The node being depended on.
    :return: The implied parameter name, or ``""`` if it has none.
    """
    label = node_key.label
    return label if _is_identifier(label) else ""


def resolve_name(text: str, root: NodeKey | None = None) -> NodeKey:
    """Resolve a name typed in the browser against the view's root.

    :param text: The name as typed. A leading ``/`` makes it absolute; anything
        else is relative to ``root``.
    :param root: The block the widget is currently rooted on, or ``None`` for
        the whole computation.
    :return: The full computation key.
    :raises GraphBuildError: If the name is blank.
    """
    text = text.strip()
    if not text:
        msg = "A node needs a name"
        raise GraphBuildError(msg)
    if root is None or text.startswith("/"):
        return to_nodekey(text)
    return to_nodekey(root).join(to_nodekey(text))


def relative_name(node_key: NodeKey, root: NodeKey | None = None) -> str:
    """Render a node's key the way :func:`resolve_name` would read it back.

    :param node_key: The node to name.
    :param root: The block the widget is currently rooted on.
    :return: A relative path when the node is inside ``root``, and an absolute
        one, marked with a leading ``/``, when it is not.
    """
    if root is None:
        return str(node_key)
    root_key = to_nodekey(root)
    inside = node_key.drop_root(root_key)
    return str(inside) if inside is not None and not inside.is_root else f"/{node_key}"


def parse_inputs(entries: Sequence[Any], root: NodeKey | None = None) -> dict[str, NodeKey]:
    """Resolve the input declarations of a calculation node.

    Each entry is either a node name, in which case the parameter is named
    after the node, or ``parameter=node``, which is what a name that cannot be
    a parameter --- or two inputs whose last path parts collide --- needs.

    :param entries: One declaration per entry. Blank entries are skipped, so
        the browser can send a textarea's lines without tidying them.
    :param root: The block the widget is currently rooted on.
    :return: Parameter name to node key, in the order declared.
    :raises GraphBuildError: If a declaration is malformed, names an unusable
        parameter, repeats one, or there are more than :data:`MAX_INPUTS`.
    """
    kwds: dict[str, NodeKey] = {}
    for entry in entries:
        text = str(entry).strip()
        if not text:
            continue
        if len(kwds) >= MAX_INPUTS:
            msg = f"A node built here may take at most {MAX_INPUTS} inputs"
            raise GraphBuildError(msg)
        parameter, separator, path = text.partition("=")
        if not separator:
            parameter, path = "", text
        node_key = resolve_name(path, root)
        parameter = parameter.strip() or default_parameter(node_key)
        if not parameter:
            msg = f"{path.strip()} cannot be a parameter name, so give it one, as in value={path.strip()}"
            raise GraphBuildError(msg)
        if not _is_identifier(parameter):
            msg = f"{parameter} is not a usable parameter name"
            raise GraphBuildError(msg)
        if parameter in kwds:
            msg = f"Two inputs both arrive as {parameter}; name one of them, as in other_{parameter}=..."
            raise GraphBuildError(msg)
        kwds[parameter] = node_key
    return kwds


def _register_source(filename: str, source: str) -> None:
    """Make ``source`` visible to :mod:`inspect` under a filename with no file.

    ``inspect.getsource`` reads through :mod:`linecache`, and consults it for a
    filename that does not exist on disk --- which is how a UI-built function
    can still show its own source in the detail panel. The ``None`` where a
    modification time belongs is what stops ``linecache.checkcache`` from
    dropping the entry when it fails to stat the file.
    """
    linecache.cache[filename] = (len(source), None, source.splitlines(keepends=True), filename)


def compile_expression(
    expression: str,
    parameters: Sequence[str],
    *,
    node_key: NodeKey,
    namespace: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Compile one expression into the function a calculation node runs.

    :param expression: Python expression, evaluated with the parameters bound
        to the values of the nodes they came from.
    :param parameters: Parameter names, in order.
    :param node_key: The node being defined, which names the function and the
        pseudo-file its source is registered under.
    :param namespace: Globals the expression is evaluated against, so it can
        use the notebook's own imports. Pass ``globals()`` for that; ``None``
        gives an empty namespace with builtins available.
    :return: The compiled function, carrying the expression it came from.
    :raises GraphBuildError: If the expression is blank, over
        :data:`MAX_EXPRESSION_LENGTH`, or does not parse as an expression.
    """
    text = expression.strip()
    if not text:
        msg = "A calculation node needs an expression"
        raise GraphBuildError(msg)
    if len(text) > MAX_EXPRESSION_LENGTH:
        msg = f"That expression is {len(text)} characters, over the limit of {MAX_EXPRESSION_LENGTH}"
        raise GraphBuildError(msg)
    filename = f"<loman node {node_key}>"
    try:
        # Checked as an expression first, so a pasted statement is refused here
        # with its own error rather than as a puzzling one about a `return`.
        compile(text, filename, "eval")
    except SyntaxError as exc:
        msg = f"That expression does not parse: {exc.msg}"
        raise GraphBuildError(msg) from exc

    func_name = node_key.label if _is_identifier(node_key.label) else _FALLBACK_FUNC_NAME
    # Wrapped in brackets and indented, so a multi-line expression stays one
    # expression: inside brackets Python joins lines and ignores indentation.
    source = f"def {func_name}({', '.join(parameters)}):\n    return (\n{indent(text, ' ' * 8)}\n    )\n"
    _register_source(filename, source)
    module_globals: dict[str, Any] = {} if namespace is None else namespace
    # Defining the function in its own locals rather than in ``module_globals``
    # keeps the notebook's namespace clean while leaving the function's globals
    # pointing at the live mapping, so it sees imports made after it was built.
    defined: dict[str, Any] = {}
    # Running text as code is what this feature is, not something it does by
    # accident, and the caller has already agreed to it: the widget only
    # reaches here with ``buildable=True``, which is off by default and
    # documented as running browser-written code in the kernel.
    exec(compile(source, filename, "exec"), module_globals, defined)  # noqa: S102 # nosec B102
    func = defined[func_name]
    # Kept so the form can be reopened on this node with what was typed in it.
    # Recovering it from the source would mean unpicking the wrapper above.
    func.__loman_expression__ = text
    return func


@dataclass(frozen=True)
class NodeDefinition:
    """A node the browser asked for, resolved against the computation."""

    key: NodeKey
    func: Callable[..., Any] | None = None
    kwds: dict[str, NodeKey] = field(default_factory=dict)
    value: Any = None
    has_value: bool = False

    def apply(self, computation: Computation) -> None:
        """Add this node to ``computation``, replacing any node of that name.

        :param computation: The computation to build in.
        """
        extra = {"value": self.value} if self.has_value else {}
        computation.add_node(self.key, self.func, kwds=dict(self.kwds) or None, **extra)


def build_definition(
    request: Mapping[str, Any],
    *,
    root: NodeKey | None = None,
    namespace: dict[str, Any] | None = None,
) -> NodeDefinition:
    """Turn one browser definition payload into a node ready to be added.

    :param request: The payload, which is untrusted: ``name``, ``kind``, and
        then ``value`` for an input node or ``inputs`` and ``expression`` for a
        calculation node.
    :param root: The block the widget is currently rooted on.
    :param namespace: Globals a compiled expression is evaluated against.
    :return: The resolved definition.
    :raises GraphBuildError: If the payload does not describe a node.
    :raises ValueWireError: If an input node's value is not a scalar this
        format supports.
    """
    node_key = resolve_name(str(request.get("name", "")), root)
    kind = request.get("kind", "input")
    if kind == "input":
        wire = request.get("value")
        if wire is None:
            return NodeDefinition(key=node_key)
        return NodeDefinition(key=node_key, value=from_wire(wire), has_value=True)
    if kind != "calc":
        msg = f"A node is either an input or a calculation, not {kind!r}"
        raise GraphBuildError(msg)
    kwds = parse_inputs(request.get("inputs") or [], root)
    func = compile_expression(str(request.get("expression", "")), list(kwds), node_key=node_key, namespace=namespace)
    return NodeDefinition(key=node_key, func=func, kwds=kwds)


def format_inputs(kwds: Mapping[str, Name], root: NodeKey | None = None) -> list[str]:
    """Render a node's parameter mapping the way :func:`parse_inputs` reads it.

    :param kwds: Parameter name to source node, as
        :meth:`Computation.get_definition_args_kwds` reports it.
    :param root: The block the widget is currently rooted on.
    :return: One declaration per input, sorted by parameter name.
    """
    entries = []
    for parameter, source in sorted(kwds.items()):
        if isinstance(source, ConstantValue):
            continue
        node_key = to_nodekey(source)
        text = relative_name(node_key, root)
        entries.append(text if default_parameter(node_key) == parameter else f"{parameter}={text}")
    return entries


def describe_definition(computation: Computation, node_key: NodeKey, root: NodeKey | None = None) -> dict[str, Any]:
    """Describe how a node is defined, so the form can be reopened on it.

    ``editable`` says whether this form can put the node back the way it found
    it. It cannot for a node that takes positional or constant arguments, which
    the form has no field for, nor for a function written in Python, whose body
    is not an expression this form could reproduce --- offering to edit one
    would mean offering to replace it with something else.

    :param computation: The computation the node belongs to.
    :param node_key: The node to describe.
    :param root: The block the widget is currently rooted on.
    :return: The payload the browser's node form reads.
    """
    func = computation.dag.nodes[node_key].get(NodeAttributes.FUNC)
    args, kwds = computation.get_definition_args_kwds(node_key)
    expression = getattr(func, "__loman_expression__", None)
    constants = [source for source in kwds.values() if isinstance(source, ConstantValue)]
    return {
        # The node's own name in the form the node form reads, which is what
        # lets the form be reopened on it while the view is rooted on a block.
        "name": relative_name(node_key, root),
        "kind": "calc" if func is not None else "input",
        "inputs": format_inputs(kwds, root),
        "expression": expression,
        "editable": not args and not constants and (func is None or expression is not None),
    }
