#!/usr/bin/env python3
"""Canonical Multi-Frame Ground-Truth Oracle Dataset Generator compliant with Protocol v1.

Loads 100% configuration from Protocol v1 (configs/protocol_v1.yaml), executes multi-seed
provenance across protocol seeds [42, 43, 44, 45, 46], computes observed update frequencies
and attribution visibility, enforces single-Gaussian intervention semantics, and locks
Global Delta Q as primary scientific label.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.generate_phase3_oracle_dataset import main

if __name__ == '__main__':
    main()
