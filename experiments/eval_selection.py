#!/usr/bin/env python3
"""Thin wrapper redirecting to experiments/evaluate_selection.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.evaluate_selection import main

if __name__ == "__main__":
    main()
