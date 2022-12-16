"""
.. include:: ../../README.md
"""

import logging

import pkg_resources

# Fetches the version of the package as defined in pyproject.toml
__version__ = pkg_resources.get_distribution("scandi_reddit").version


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] <%(name)s> %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
