"""Runtime configuration helpers.

Centralizes environment tweaks and compatibility patches that make the CLI
scripts in this repo more robust across dependency versions.
"""

from __future__ import annotations

import os


def configure_runtime() -> None:
    """Apply environment defaults and compatibility patches (best-effort)."""

    # Environment hygiene: avoid importing TF/JAX via Transformers and keep logs quiet.
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    # Some audio stacks still require the pure-Python protobuf implementation.
    os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

    _patch_protobuf_message_factory()


def _patch_protobuf_message_factory() -> None:
    """Compatibility shim for protobuf>=5 where MessageFactory.GetPrototype was removed."""

    try:
        from google.protobuf import message_factory as _message_factory  # type: ignore
    except Exception:
        return

    try:
        if hasattr(_message_factory.MessageFactory, "GetPrototype"):
            return
        get_message_class = getattr(_message_factory, "GetMessageClass", None)
        if get_message_class is None:
            return

        def _get_prototype(self, descriptor):  # type: ignore[no-untyped-def]
            return get_message_class(descriptor)

        setattr(_message_factory.MessageFactory, "GetPrototype", _get_prototype)
    except Exception:
        return

