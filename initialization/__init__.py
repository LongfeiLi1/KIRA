"""Deterministic, specification-only active-filter synthesis; no simulator or API.

The receipt is an initialization prediction, never physical acceptance. Case IDs
are labels, not lookup keys. Historical benchmark seeds are not inputs.
"""
from .synthesis import InitializationError, synthesize, canonical, digest
__all__ = ['InitializationError', 'synthesize', 'canonical', 'digest']
