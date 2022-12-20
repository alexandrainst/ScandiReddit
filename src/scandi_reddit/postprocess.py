"""Post-process the built corpus."""

import json
import logging
import re
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from .utils import ACCEPTABLE_SUBREDDITS, DRUG_SUBREDDITS, NSFW_SUBREDDITS

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
    corpus = remove_inappropriate_comments(corpus=corpus)
    logger.info(
        f"Removed {prev_count - len(corpus):,} comments that contained too little "
        "content."
    )

    # Save the corpus
    output_path = path.parent / f"{path.stem}{suffix}.jsonl"
    with output_path.open("w") as f:
        for _, row in corpus.iterrows():
            f.write(json.dumps(row.to_dict()) + "\n")


def remove_inappropriate_comments(corpus: pd.DataFrame) -> pd.DataFrame:
    """Remove the inappropriate comments.

    Args:
        corpus (pd.DataFrame):
            The corpus.

    Returns:
        pd.DataFrame:
            The corpus without the inappropriate comments.
    """
    # Define the lists of banned and not banned subreddits
    banned_subreddits = NSFW_SUBREDDITS + DRUG_SUBREDDITS
    not_banned_subreddits = ACCEPTABLE_SUBREDDITS

    # Filter the corpus into the comments within the banned and not banned subreddits
    banned_df = corpus[corpus.subreddit.isin(banned_subreddits)]
    not_banned_df = corpus[corpus.subreddit.isin(not_banned_subreddits)]

    # Create the part of the corpus to be used for training and evaluation
    train_test_df = pd.concat([banned_df, not_banned_df])
    train_test_df["label"] = train_test_df.copy().subreddit.isin(banned_subreddits)

    # Split the corpus into train and test sets
    train, test = train_test_split(
        train_test_df,
        test_size=min(int(len(train_test_df) * 0.1), 1_000),
        stratify=train_test_df.label,
        random_state=4242,
    )

    # Shuffle the training split
    train = train.sample(frac=1.0, random_state=4242)

    # Define the classifier
    clf = make_pipeline(
        TfidfVectorizer(
            ngram_range=(1, 4),
            lowercase=True,
            max_features=60_000,
            norm=None,
            analyzer="char",
        ),
        LogisticRegression(
            class_weight="balanced",
            max_iter=10_000,
            random_state=4242,
            C=0.1,
        ),
    )

    # Train the model
    logger.info("Training classifier to detect inappropriate comments.")
    clf.fit(train.doc, train.label)

    # Evaluate the model on the test set
    predictions = clf.predict(test.doc)
    f1 = f1_score(predictions, test.label)
    recall = recall_score(predictions, test.label)
    precision = precision_score(predictions, test.label)

    # Log the scores
    logger.info(f"F1-score on the test dataset: {f1:.2%}")
    logger.info(f"Recall score on the test dataset: {recall:.2%}")
    logger.info(f"Precision score on the test dataset: {precision:.2%}")

    # Predict the labels for the entire corpus
    predictions = clf.predict(corpus.doc)

    # Remove the comments that were predicted to be inappropriate
    corpus = corpus[~predictions]

    # Remove all the comments belonging to the banned subreddits
    corpus = corpus[~corpus.subreddit.isin(banned_subreddits)]

    # Return the corpus
    return corpus
