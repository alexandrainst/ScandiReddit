"""Builds a Scandinavian Reddit dataset."""

import json
import logging
import subprocess
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

import pandas as pd
import zstandard
from datasets.arrow_dataset import Dataset
from joblib import Parallel, delayed
from nlp_dedup import Deduper
from tqdm.auto import tqdm

from scandi_reddit.postprocess import postprocess

from .download import download_reddit_file
from .language_filter import filter_comment

# Set up logging
logger = logging.getLogger(__name__)


def build_reddit_dataset(
    overwrite: bool = False,
    n_jobs: int = -2,
    starting_year: int = 2005,
    starting_month: int = 1,
    skip_download: bool = False,
    hub_repo_id: Optional[str] = None,
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
        skip_download (bool, optional):
            Whether to skip downloading the files. If this is set then the "data/raw"
            directory must contain the files "reddit-da.jsonl", "reddit-no.jsonl",
            "reddit-sv.jsonl" and "reddit-is.jsonl". Defaults to False.
        hub_repo_id (Optional[str], optional):
            The ID of the Hugging Face Hub repository to upload the dataset to. If
            this is set then the dataset will be uploaded to the Hugging Face Hub.
            If None then the dataset will not be uploaded. Defaults to None.
    """
    # Set up paths to data directories
    raw_data_dir = Path("data") / "raw"
    processed_data_dir = Path("data") / "processed"
    final_data_dir = Path("data") / "final"

    # Create language mapping
    language_mapping = {
        "da": "Danish",
        "sv": "Swedish",
        "no": "Norwegian",
        "is": "Icelandic",
    }

    # Set up the output files
    output_paths = {
        lang: processed_data_dir / f"reddit-{lang}.jsonl"
        for lang in language_mapping.keys()
    }

    # Ensure `n_jobs` is non-negative
    if n_jobs < 0:
        n_jobs = cpu_count() + n_jobs + 1

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

    # Download the Reddit dumps and apply the language filter
    if not skip_download:

        logger.info(f"Fetching Reddit comments using {n_jobs} jobs in parallel.")

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

    # Post-process the files
    for lang, path in output_paths.items():
        logger.info(f"Post-processing the {language_mapping[lang]} corpus.")
        postprocess(path=path, suffix="-postprocessed")

    # Initialise the Deduper
    deduper = Deduper(
        split_method="word_ngram",
        num_minhashes=128,
        ngram_size=5,
        similarity_threshold=0.8,
        batch_size=1_000_000,
        n_jobs=n_jobs,
        random_seed=4242,
        store_config_to_disk=True,
        store_mask_to_disk=True,
        store_lsh_cache_to_disk=False,
        store_corpus_to_disk=False,
    )

    # Create the corpus generator
    def build_corpus() -> Generator[str, None, None]:
        for path in output_paths.values():
            path_processed = path.parent / f"{path.stem}-postprocessed.jsonl"
            with path_processed.open() as f:
                for line in f:
                    line = json.loads(line)
                    yield line["doc"]  # type: ignore[index]

    # Count the lines in the corpus
    num_docs = 0
    for path in output_paths.values():
        proc = subprocess.Popen(["wc", "-l", str(path)], stdout=subprocess.PIPE)
        num_docs += int(proc.communicate()[0].decode().split()[0])

    # Deduplicate the files
    deduper.deduplicate(
        corpus=build_corpus(),
        output_dir=processed_data_dir / "deduplicated",
        num_docs=num_docs,
        overwrite=True,
    )

    # Load the deduplication mask
    mask_path = processed_data_dir / "deduplicated" / "mask.jsonl"
    with mask_path.open() as f:
        mask = [json.loads(line) for line in f]

    # Load all the deduplicated files
    all_records: List[Dict[str, Any]] = list()
    idx: int = 0
    for path in output_paths.values():
        path_processed = path.parent / f"{path.stem}-postprocessed.jsonl"
        with path_processed.open() as f:
            for line in f:
                if not mask[idx]["duplicate"]:
                    record = json.loads(line)
                    all_records.append(record)
                idx += 1

    # Convert the records to a Hugging Face dataset
    df = pd.DataFrame.from_records(all_records)
    dataset = Dataset.from_pandas(df)

    # Save the dataset to disk
    dataset.save_to_disk(str(final_data_dir / "scandireddit"))

    # Push the dataset to the Hugging Face Hub
    if hub_repo_id is not None:
        dataset.push_to_hub(hub_repo_id)


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
        desc=f"Processing comments from {input_path.name}",
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

        # Split the batch into individual comments
        comments = batch.splitlines()

        # Process the comments in parallel
        with Parallel(n_jobs=n_jobs) as parallel:
            records = parallel(
                delayed(filter_comment)(comment) for comment in comments[:-1]
            )

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
        buffer = comments[-1]

    # Close the progress bar
    progress_bar.close()

    # Close the output files
    for output_file in output_files.values():
        output_file.close()

    # Close the file
    f.close()


if __name__ == "__main__":
    build_reddit_dataset()
