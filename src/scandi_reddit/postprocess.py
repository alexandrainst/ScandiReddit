"""Post-process the built corpus."""

import json
import logging
import re
from pathlib import Path
from typing import List, Union

import pandas as pd

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
    banned_subreddits = get_banned_subreddits(
        corpus.subreddit.unique()
    )  # NSFW_SUBREDDITS + DRUG_SUBREDDITS
    corpus = corpus[~corpus.subreddit.isin(banned_subreddits)]
    logger.info(f"Removed {prev_count - len(corpus):,} inappropriate comments.")

    # Save the corpus
    output_path = path.parent / f"{path.stem}{suffix}.jsonl"
    with output_path.open("w") as f:
        for _, row in corpus.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")


def get_banned_subreddits(subreddits: List[str]) -> List[str]:
    """Check if a list of subreddits are banned.

    Args:
        subreddits (List[str]):
            The list of subreddits to check.

    Returns:
        List[str]:
            The list of banned subreddits.

    Raises:
        ValueError:
            If the list of subreddits is empty.
    """
    banned_words = [
        "nsfw",
        "gonewild",
        "cock" "tits" "titties",
        "milf",
        "porn",
        "dirty",
        "fraek",
        "nipple",
        "trusse",
        "buksebule",
        "rape",
        "jodel",
        "weed",
        "drugs",
        "droger",
        "stoffer",
        "darknet",
        "sortemarked",
        "psyches",
        "rusmidler",
        "naket",
    ]

    # Filter the subreddits
    banned_subreddits = [
        subreddit
        for subreddit in subreddits
        if any(keyword in subreddit.lower() for keyword in banned_words)
    ]

    return banned_subreddits
