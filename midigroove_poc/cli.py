"""Unified CLI dispatcher for this repo.

This keeps the original entrypoints intact while offering a single, clearer
entrypoint:

  python -m midigroove_poc drumgrid ...
  python -m midigroove_poc expressivegrid ...
  python -m midigroove_poc eval ...
"""

from __future__ import annotations

import sys
from typing import Optional


def main(argv: Optional[list[str]] = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "usage: python -m midigroove_poc <command> ...\n\n"
            "commands:\n"
            "  drumgrid       build caches with codec tokens\n"
            "  expressivegrid train/predict expressivegrid -> codec tokens\n"
            "  eval           evaluate checkpoints across codecs\n\n"
            "examples:\n"
            "  python -m midigroove_poc drumgrid train --help\n"
            "  python -m midigroove_poc expressivegrid train --help\n"
            "  python -m midigroove_poc eval --help\n"
        )
        return

    cmd, rest = argv[0], argv[1:]

    if cmd == "drumgrid":
        from .drumgrid import main as drumgrid_main

        return drumgrid_main(rest)

    if cmd == "expressivegrid":
        from .expressivegrid import main as expressivegrid_main

        return expressivegrid_main(rest)

    if cmd == "eval":
        from .eval import main as eval_main

        return eval_main(rest)

    raise SystemExit(f"unknown command {cmd!r} (expected: drumgrid|expressivegrid|eval)")


if __name__ == "__main__":
    main()
