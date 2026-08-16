"""Serialization profiles: how much of a saved computation is readable text.

A profile and a container are independent choices. The profile decides how a
*value* is encoded --- inline JSON numbers, or bytes written out of line. The
container decides where those bytes *land* --- one text document, a zip, or a
directory. Collapsing the two into a single setting looks tidier and immediately
fails on real cases: a readable manifest inside a zip is genuinely useful, and
``dumps()`` returning a string can only ever be readable-plus-single-document.

Only one combination is impossible: the efficient profile has nowhere to put
blobs in a single JSON document, and base64-inlining them would inflate by a
third and force a read-all. That case raises and points at ``container="zip"``.
"""

from __future__ import annotations

import fnmatch
from typing import Any

import attrs

from .compression import DEFAULT_COMPRESSION

# Below this, a separate container member --- its own entry, filename and seek
# --- costs more than the JSON it saves, and keeping small values inline is what
# preserves "open the manifest and read it". 8 KiB is about a 1024-element
# float64 array.
DEFAULT_INLINE_MAX_BYTES = 8 * 1024


@attrs.frozen
class SerializationProfile:
    """How values are encoded for one save.

    :ivar name: Identifier recorded in the manifest.
    :ivar inline_max_bytes: Values estimated at or below this many bytes stay
        inline. ``None`` keeps everything inline, whatever the container.
    :ivar overrides: Selector-to-settings map, letting one save treat some nodes
        differently. A selector is a node-key glob (``"market_data/**"``) or a
        tag (``"tag:raw"``).
    """

    name: str
    inline_max_bytes: int | None = None
    #: ``"none"``, or a codec and optional level such as ``"zstd:1"`` or
    #: ``"zlib:6"``. Named, never inferred: a saved file should compress the way
    #: you asked, not the way something guessed. Whichever of the compressed and
    #: raw payloads is smaller is what gets stored.
    compression: str = "none"
    dedupe: str = "identity"
    checksums: bool = False
    frame_encoding: str = "npy"
    overrides: dict[str, dict[str, Any]] = attrs.field(factory=dict)

    def wants_blob(self, nbytes: int | None) -> bool:
        """Return whether a value of *nbytes* should be written out of line.

        A transformer that cannot estimate its size passes ``None`` and is taken
        at its word that the value is worth storing out of line.
        """
        if self.inline_max_bytes is None:
            return False
        if nbytes is None:
            return True
        return nbytes > self.inline_max_bytes

    def settings_for(self, node: str | None, tags: frozenset[str] = frozenset()) -> dict[str, Any]:
        """Return the override settings that apply to *node*.

        Later matches win, so a more specific selector listed after a general one
        takes precedence --- the order the overrides were written in.
        """
        settings: dict[str, Any] = {}
        for selector, values in self.overrides.items():
            if _selector_matches(selector, node, tags):
                settings.update(values)
        return settings


def _selector_matches(selector: str, node: str | None, tags: frozenset[str]) -> bool:
    """Return whether *selector* applies to a node with *tags*."""
    if selector.startswith("tag:"):
        return selector[4:] in tags
    if node is None:
        return False
    # fnmatch treats "*" as matching separators too, so "a/**" and "a/*" both
    # match nested keys. Node keys are shallow in practice; exactness here would
    # mean a path-glob implementation for no observed benefit.
    return fnmatch.fnmatchcase(node, selector)


#: Everything inline: the file is JSON you can open and read end to end.
READABLE = SerializationProfile(name="readable", inline_max_bytes=None)

#: Large values out of line as binary blobs; small ones stay inline, so the
#: manifest still describes every value's shape without decoding anything.
#: Compressed with zstd at level 1, which on measured data gives around 9x on a
#: realistic price series and rejects incompressible data at about 1 GB/s. A
#: blob that does not shrink is stored raw, so the cost of the attempt is
#: bounded and nothing is ever stored larger than it started.
EFFICIENT = SerializationProfile(
    name="efficient",
    inline_max_bytes=DEFAULT_INLINE_MAX_BYTES,
    compression=DEFAULT_COMPRESSION,
)

_BY_NAME = {p.name: p for p in (READABLE, EFFICIENT)}


def resolve_profile(profile: str | SerializationProfile | None) -> SerializationProfile:
    """Return a :class:`SerializationProfile` from a name, an instance or ``None``."""
    if profile is None:
        return EFFICIENT
    if isinstance(profile, SerializationProfile):
        return profile
    try:
        return _BY_NAME[profile]
    except KeyError:
        msg = f"Unknown profile {profile!r}; expected one of {sorted(_BY_NAME)} or a SerializationProfile"
        raise ValueError(msg) from None
