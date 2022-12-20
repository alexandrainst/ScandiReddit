# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [Unreleased]
### Added
- Added postprocessing of corpora, including removal of duplicates, bot comments, and
  inappropriate comments, where the latter is using a list of subreddits along with a
  classifier, to maximise recall.


## [v0.1.0] - 2022-12-20
### Added
- Initial release, which includes the CLI command `build`, which builds the
  Scandinavian Reddit corpus. Run `build --help` to see more information.
