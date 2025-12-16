from collections.abc import Iterable
from typing import TypeVar

from tqdm import tqdm

T = TypeVar("T")


def progress(
    iterable: Iterable[T],
    verbose: bool,
    desc: str,
    *,
    total: int | None = None,
    unit: str | None = None,
) -> Iterable[T]:
    """Wrap an iterable with a TQDM progress bar when verbose output is enabled."""

    if not verbose:
        return iterable
    return tqdm(iterable, desc=desc, total=total, unit=unit)
