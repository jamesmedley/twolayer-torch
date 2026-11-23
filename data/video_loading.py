"""Load raw natural-movie chunks from the local data directory.

The raw files are a sequence of big-endian float32 files, each containing a
``128 x 128 x 64`` chunk stored in column-major order. They are converted here to a NumPy array of shape
``(n_chunks, n_frames, height, width)``.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

_CHUNK_RE = re.compile(r"chunk(\d+)$")


def _chunk_number(path: Path) -> int | None:
    match = _CHUNK_RE.fullmatch(path.name)
    return int(match.group(1)) if match is not None else None


def find_chunk_dir(data_root: str | Path) -> Path:
    """Return the directory containing ``chunk1``, ``chunk2``, ...

    Both ``<data_root>/vid075-chunks`` and ``<data_root>`` are checked.
    """
    root = Path(data_root)
    candidates = (root / "vid075-chunks", root)

    for candidate in candidates:
        if not candidate.is_dir():
            continue
        if any(
            path.is_file() and _chunk_number(path) is not None
            for path in candidate.iterdir()
        ):
            return candidate

    checked = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(
        "Could not find any chunk files. Expected chunk1, chunk2, ... in one "
        f"of: {checked}. Put the raw chunks in {root / 'vid075-chunks'} or "
        "pass a different --data-root."
    )


def load_raw_dataset(
    data_root: str | Path,
    *,
    n_chunks: int | None = None,
    frames_per_chunk: int = 64,
    frame_size: int = 128,
    verbose: bool = True,
) -> np.ndarray:
    """Load raw movie chunks as ``(n_chunks, n_frames, height, width)``.

    Parameters
    ----------
    data_root:
        Directory containing ``vid075-chunks/`` or the chunk files directly.
    n_chunks:
        Optional limit on the number of chunks to read, useful for smoke tests.
    frames_per_chunk, frame_size:
        Raw chunk dimensions. The defaults match the natural-movie dataset.
    """
    if n_chunks is not None and n_chunks <= 0:
        raise ValueError(f"n_chunks must be positive, got {n_chunks}.")

    chunk_dir = find_chunk_dir(data_root)
    paths = sorted(
        (
            path
            for path in chunk_dir.iterdir()
            if path.is_file() and _chunk_number(path) is not None
        ),
        key=_chunk_number,
    )
    if n_chunks is not None:
        paths = paths[:n_chunks]

    if not paths:
        raise FileNotFoundError(f"No chunks found in {chunk_dir}.")

    expected_values = frames_per_chunk * frame_size * frame_size
    data = np.empty(
        (len(paths), frames_per_chunk, frame_size, frame_size),
        dtype=np.float32,
    )

    for index, path in enumerate(paths):
        values = np.fromfile(path, dtype=">f4")
        if values.size != expected_values:
            raise ValueError(
                f"{path} contains {values.size} values, expected "
                f"{expected_values}."
            )
        chunk = values.reshape(
            (frame_size, frame_size, frames_per_chunk), order="F"
        )
        data[index] = chunk.transpose(2, 0, 1)

    if verbose:
        print(
            f"Loaded {len(paths)} raw chunks from {chunk_dir}: "
            f"shape={data.shape}, mean={data.mean():.4f}, std={data.std():.4f}"
        )

    return data
