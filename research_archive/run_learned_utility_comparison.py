#!/usr/bin/env python3
"""Thin wrapper redirecting to evaluate_utility_model.py for learned utility comparison."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.evaluate_utility_model import main

if __name__ == "__main__":
    main()
