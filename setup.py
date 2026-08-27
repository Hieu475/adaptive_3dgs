"""
Python package setup.
"""
from setuptools import setup, find_packages

setup(
    name="adaptive_3dgs",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyyaml",
    ],
)
