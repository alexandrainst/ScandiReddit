# ScandiReddit

Construction of a Scandinavian Reddit dataset.

______________________________________________________________________
[![Documentation](https://img.shields.io/badge/docs-passing-green)](https://alexandrainst.github.io/ScandiReddit/scandi_reddit.html)
[![License](https://img.shields.io/github/license/alexandrainst/ScandiReddit)](https://github.com/alexandrainst/ScandiReddit/blob/main/LICENSE)
[![LastCommit](https://img.shields.io/github/last-commit/alexandrainst/ScandiReddit)](https://github.com/alexandrainst/ScandiReddit/commits/main)
[![Code Coverage](https://img.shields.io/badge/Coverage-0%25-red.svg)](https://github.com/alexandrainst/ScandiReddit/tree/main/tests)


Developers:

- Dan Saattrup Nielsen (dan.nielsen@alexandra.dk)


## Setup

### Set up the environment

1. Run `make install`, which installs Poetry (if it isn't already installed), sets up a virtual environment and all Python dependencies therein.
2. Run `source .venv/bin/activate` to activate the virtual environment.

### Install new packages

To install new PyPI packages, run:

```
$ poetry add <package-name>
```

### Auto-generate API documentation

To auto-generate API document for your project, run:

```
$ make docs
```

To view the documentation, run:

```
$ make view-docs
```

## Tools used in this project
* [Poetry](https://towardsdatascience.com/how-to-effortlessly-publish-your-python-package-to-pypi-using-poetry-44b305362f9f): Dependency management
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatically create an API documentation for your project

## Project structure
```
.
├── .flake8
├── .github
│   └── workflows
│       ├── ci.yaml
│       └── docs.yaml
├── .gitignore
├── .pre-commit-config.yaml
├── CHANGELOG.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── LICENSE
├── README.md
├── data
├── makefile
├── models
├── notebooks
├── poetry.toml
├── pyproject.toml
├── src
│   ├── scandi_reddit
│   │   ├── __init__.py
│   │   ├── build.py
│   │   ├── cli.py
│   │   ├── download.py
│   │   └── language_filter.py
│   └── scripts
│       ├── fix_dot_env_file.py
│       └── versioning.py
└── tests
    └── __init__.py
```
