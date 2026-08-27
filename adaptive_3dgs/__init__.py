import torch

try:
    from . import _C
except ImportError:
    _C = None
