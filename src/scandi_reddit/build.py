"""Builds a Scandinavian Reddit dataset."""

import logging
from multiprocessing import cpu_count
from pathlib import Path

import zstandard
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from .download import download_reddit_file
from .filter import filter_comment

# Set up logging
logger = logging.getLogger(__name__)


def build_reddit_dataset(
    overwrite: bool = False,
    n_jobs: int = -2,
    starting_year: int = 2005,
    starting_month: int = 1,
) -> None:
    """Build a Scandinavian Reddit dataset.

    Args:
        overwrite (bool, optional):
            Whether to overwrite existing files. Defaults to False.
        n_jobs (int, optional):
            The number of jobs to run in parallel. Can be set to a negative number to
            use all but that number of cores. Defaults to -2.
        starting_year (int, optional):
            The year to start downloading from. Defaults to 2005.
        starting_month (int, optional):
            The month to start downloading from. Defaults to 1.
    """
    # Ensure `n_jobs` is non-negative
    if n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

    logger.info(f"Fetching Reddit posts using {n_jobs} jobs in parallel.")

    # Set up the output files
    raw_data_dir = Path("data") / "raw"
    output_paths = {
        lang: raw_data_dir / f"reddit_{lang}.jsonl" for lang in ["da", "sv", "no", "is"]
    }

    # Remove the previous files if `overwrite` is set
    if overwrite:
        for path in output_paths.values():
            path.unlink(missing_ok=True)

    # Replace starting year and month by the newest file present in the raw data
    # folder, if any
    existing_files = list(raw_data_dir.glob("RC_*.zst"))
    for file in existing_files:
        year = int(file.stem.split("_")[1].split("-")[0])
        month = int(file.stem.split("_")[1].split("-")[1])
        starting_year = max(starting_year, year)
        starting_month = max(starting_month, month)

    for year in range(starting_year, 2030):
        for month in range(starting_month, 13):

            # Download the file
            input_path = download_reddit_file(year=year, month=month)

            # If the download failed then skip to the next month
            if not input_path.exists():
                continue

            # Extract the comments from the file
            extract_comments_from_file(
                input_path=input_path,
                output_paths=output_paths,
                n_jobs=n_jobs,
            )

            # Delete the input file again
            input_path.unlink()

        # Set the starting month to 1
        starting_month = 1


def extract_comments_from_file(
    input_path: Path,
    output_paths: dict[str, Path],
    n_jobs: int,
) -> None:
    """Extract comments from a Reddit file.

    Args:
        input_path (Path):
            The path to the input file.
        output_paths (dict[str, Path]):
            The paths to the output files.
        n_jobs (int):
            The number of jobs to run in parallel.
    """
    # Open the file
    f = input_path.open("rb")

    # Open up the output files
    output_files = {
        lang: output_file.open("a") for lang, output_file in output_paths.items()
    }

    # Create a decompressor
    decompressor = zstandard.ZstdDecompressor(max_window_size=2**31)

    # Create a stream reader
    stream_reader = decompressor.stream_reader(f)

    # Initialise the buffer
    buffer: str = ""

    # Create progress bar, with unit being millions
    progress_bar = tqdm(
        desc=f"Processing posts from {input_path.name}",
        unit_scale=True,
    )

    # Infinite loop, break when we reach the end of the file
    while True:

        # Load a batch of data, break if it cannot be loaded
        try:
            batch = stream_reader.read(1_000_000_000)
        except zstandard.ZstdError:
            logger.debug("Could not load batch.")
            break

        # Decode the batch, skip if it cannot be decoded
        try:
            batch = batch.decode()
        except UnicodeDecodeError:
            logger.debug(f"Could not decode batch from {input_path.name}")
            continue

        # Break if we reached the end of the file
        if not batch:
            logger.debug(f"Reached end of file {input_path.name}")
            break

        # Add the buffer
        batch = buffer + batch

        # Split the batch into individual posts
        posts = batch.splitlines()

        # Process the posts in parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            records = parallel(delayed(filter_comment)(post) for post in posts[:-1])

        # If `records` is None then skip to the next file
        if records is None:
            logger.debug(f"No records found in {input_path.name}")
            continue

        # Iterate over the records, writing them to the output files
        for item in records:

            # Skip if the record is None
            if item is None:
                progress_bar.update()
                continue

            # Unpack the record
            record, lang = item

            # Write the record to the correct file
            if lang in output_files:
                output_files[lang].write(record + "\n")

            # Up the progress bar
            progress_bar.update()

        # Update the buffer
        buffer = posts[-1]

    # Close the progress bar
    progress_bar.close()

    # Close the output files
    for output_file in output_files.values():
        output_file.close()

    # Close the file
    f.close()
