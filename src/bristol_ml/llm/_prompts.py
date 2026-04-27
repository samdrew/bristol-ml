"""Stage 14 — prompt loading + SHA-256 versioning.

Plan §1 D7 / D10 / NFR-5: the active prompt is a plain-text file under
``conf/llm/prompts/`` (no Jinja). The first 12 hex chars of the
SHA-256 digest of the file's bytes is recorded in every
:class:`~bristol_ml.llm.ExtractionResult` so a swap of the prompt
file produces a different hash — *"we swapped the prompt and
everything changed"* is then diagnosable from the output.

Why a hash and not the filename or a version field?

- The filename is human-readable but mutable; ``extract_v1.txt`` can
  silently drift if the file is edited in place.
- A version field in the file would solve that, but adds a parsing
  contract on top of "the file is plain text".
- A bytes hash is the cheapest content-derived identity that survives
  edits and renames; collisions over the project lifetime are
  vanishing (12 hex chars = 48 bits = 1 in 281 trillion).

The 12-char prefix is plan-§5 prescribed; if a future stage needs
the full SHA-256 digest (e.g. for joining onto an external prompt
registry) it can be reconstructed from the same input bytes via
``hashlib.sha256(prompt_bytes).hexdigest()``.

Usage:

    text, prompt_hash = load_prompt(Path("conf/llm/prompts/extract_v1.txt"))
    # text is the prompt body to send to the LLM;
    # prompt_hash is the 12-hex-char identity to stamp on results.

The function deliberately does **not** template the ``{event_json}``
placeholder — that is the LLM extractor's responsibility (T4) and
keeps this module purely about hashing + I/O.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

__all__ = [
    "PROMPT_HASH_PREFIX_CHARS",
    "load_prompt",
    "prompt_sha256_prefix",
]


# Plan §5 schema sketch: ``ExtractionResult.prompt_hash`` is the first
# 12 hex chars of the SHA-256 digest of the prompt file's bytes. The
# constant is exported so tests + downstream consumers can reference
# it rather than hard-coding ``12``.
PROMPT_HASH_PREFIX_CHARS = 12


def prompt_sha256_prefix(prompt_bytes: bytes) -> str:
    """Return the first :data:`PROMPT_HASH_PREFIX_CHARS` chars of the SHA-256.

    Plan §5: ``ExtractionResult.prompt_hash`` carries the truncated
    form so the field is short enough to read inline in stdout output
    and parquet preview frames.
    """
    return hashlib.sha256(prompt_bytes).hexdigest()[:PROMPT_HASH_PREFIX_CHARS]


def load_prompt(path: Path) -> tuple[str, str]:
    """Read a prompt file and return ``(text, prompt_hash)``.

    The hash is computed on the *raw bytes* of the file, not on the
    decoded text — this keeps two files with identical content but
    different line endings hashed identically only if the byte
    representation is identical, which is the strict reading of
    "any change to the file produces a different hash" from NFR-5.

    Encoding is fixed to UTF-8; the project's coding-conventions
    section names UTF-8 as the default. A non-UTF-8 prompt file is a
    setup error and surfaces as :class:`UnicodeDecodeError` at this
    boundary rather than later when the LLM call fails on unknown
    characters.

    Raises :class:`FileNotFoundError` (with the path named) if the
    file is missing — which the configured ``LlmExtractorConfig.prompt_file``
    must point at.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found at {path}. "
            "Set LlmExtractorConfig.prompt_file to a valid path "
            "(default: conf/llm/prompts/extract_v1.txt)."
        )
    raw = path.read_bytes()
    text = raw.decode("utf-8")
    return text, prompt_sha256_prefix(raw)
