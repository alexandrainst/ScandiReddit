"""Post-process the built corpus."""

import json
import logging
import re
from pathlib import Path
from typing import Union

import pandas as pd

from .utils import DRUG_SUBREDDITS, NSFW_SUBREDDITS

# Set up logging
logger = logging.getLogger(__name__)


def postprocess(path: Union[str, Path], suffix: str = "-postprocessed") -> None:
    """Post-process the built corpus.

    Args:
        path (str or Path):
            The path to the corpus file.
        suffix (str, optional):
            The suffix to append to the output file. Defaults to "-postprocessed".
    """
    # Convert the path to a Path object
    path = Path(path)

    # Load the corpus as a Pandas DataFrame
    with path.open() as f:
        records = [json.loads(line) for line in f]
        corpus = pd.DataFrame.from_records(records)

    # Remove the duplicates
    prev_count = len(corpus)
    corpus = corpus.drop_duplicates(subset="doc")
    if corpus is None:
        raise ValueError("The corpus is empty.")
    logger.info(f"Removed {prev_count - len(corpus):,} duplicate comments.")

    # Remove the comments writted by bots
    prev_count = len(corpus)
    corpus = corpus[~corpus.doc.str.contains("I am a bot")]
    logger.info(f"Removed {prev_count - len(corpus):,} bot comments.")

    # Remove the comments with less than 20 characters and spaces
    prev_count = len(corpus)
    corpus = corpus[
        corpus.doc.map(lambda doc: len(re.sub(r"[^a-zæøå ]", "", doc.lower())) > 20)
    ]
    logger.info(
        f"Removed {prev_count - len(corpus):,} comments that contained too little "
        "content."
    )

    # Remove the inappropriate comments
    prev_count = len(corpus)
    banned_subreddits = NSFW_SUBREDDITS + DRUG_SUBREDDITS
    corpus = corpus[~corpus.subreddit.isin(banned_subreddits)]
    logger.info(
        f"Removed {prev_count - len(corpus):,} comments from banned subreddits."
    )

    # Save the corpus
    output_path = path.parent / f"{path.stem}{suffix}.jsonl"
    with output_path.open("w") as f:
        for _, row in corpus.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")
