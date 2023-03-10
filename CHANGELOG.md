# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/) and this
project adheres to [Semantic Versioning](http://semver.org/spec/v2.0.0.html).


## [v0.2.1] - 2022-12-21
### Fixed
- Some of the banned words were not banned correctly - these are now correctly removed.


## [v0.2.0] - 2022-12-21
### Added
- Added postprocessing of corpora, including removal of duplicates, bot comments, and
  removing comments from inappropriate subreddits.
- Added `--hub-repo-id` to the CLI, which can be used to upload the resulting dataset
  to the Hugging Face Hub.


## [v0.1.0] - 2022-12-20
### Added
- Initial release, which includes the CLI command `build`, which builds the
  Scandinavian Reddit corpus. Run `build --help` to see more information.
