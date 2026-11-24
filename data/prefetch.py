"""Background prefetching for clip streams.

Wraps any clip stream (``BSD500ClipStream`` or ``Layer1ClipStream``) and
prepares one batch ahead on a background thread so CPU data preparation
overlaps with GPU compute.
"""

from __future__ import annotations

from queue import Queue
from threading import Thread

import torch


class PrefetchStream:
    """Prefetch one batch ahead of the training loop.

    Wraps a clip stream and calls its ``sample_batch`` method on a background
    thread. The training loop calls :meth:`prefetch` to enqueue the next batch
    and :meth:`fetch` to consume it, overlapping CPU data preparation with GPU
    compute. At most one batch is prefetched at a time.
    """

    def __init__(self, stream) -> None:
        self.stream = stream
        self._queue: Queue = Queue(maxsize=1)
        self._thread: Thread | None = None
        self._exception: BaseException | None = None

    def prefetch(self, n_clips: int) -> None:
        """Prepare ``stream.sample_batch(n_clips)`` on a background thread.

        Joins any previously started prefetch thread first (so at most one
        batch is in flight), then starts a new thread.
        """
        if self._thread is not None:
            self._thread.join()
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc

        def _target() -> None:
            try:
                batch = self.stream.sample_batch(n_clips)
            except BaseException as exc:  # noqa: BLE001
                self._exception = exc
                return
            self._queue.put(batch)

        self._thread = Thread(target=_target, daemon=True)
        self._thread.start()

    def fetch(self) -> torch.Tensor:
        """Return the prefetched batch, blocking until it is ready."""
        if self._thread is not None:
            self._thread.join()
        if self._exception is not None:
            exc = self._exception
            self._exception = None
            raise exc
        return self._queue.get()

    def close(self) -> None:
        """Join any in-flight prefetch thread."""
        if self._thread is not None:
            self._thread.join()
            self._thread = None
