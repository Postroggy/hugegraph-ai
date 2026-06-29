"""Offline entity disambiguation for car graph extraction results."""

__all__ = ["run_disambiguation"]


def run_disambiguation(*args, **kwargs):
    from .run import run_disambiguation as _run_disambiguation

    return _run_disambiguation(*args, **kwargs)
