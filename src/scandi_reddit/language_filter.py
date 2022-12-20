"""Filter downloaded Reddit files."""

import json
import logging
from typing import Tuple, Union

from luga import language as detect_language

# Set up logger
logger = logging.getLogger(__name__)


def filter_comment(comment: str) -> Union[Tuple[str, str], None]:
    """Process a comment.

    Args:
        comment (str):
            The comment to process.

    Returns:
        pair of str, or None:
            The comment and the language, or None if the comment is not valid.
    """
    # Load the comment, skip if it cannot be loaded
    try:
        dct = json.loads(comment)
    except json.JSONDecodeError:
        logger.debug(f"Could not load comment: {comment}")
        return None

    # Extract the information, skip if it cannot be extracted
    try:
        doc = dct["body"]
        subreddit = dct["subreddit"]
    except KeyError:
        logger.debug(f"No body or subreddit in {dct}")
        return None

    # If the comment is too short, skip it
    if len(doc) < 20:
        logger.debug(
            f"Length of comment at {len(doc):,} is under the 20 character limit."
        )
        return None

    # Get the language of the comment
    lang = detect_language(doc.replace("\n", " "))

    # If the language certainty is too low, skip the comment
    if lang.score < 0.7:
        logger.debug(
            f"The language certainty at {lang.score:.2%} was under the 70% threshold."
        )
        return None

    # Store record
    record = dict(
        doc=doc,
        subreddit=subreddit,
        language=lang.name,
        language_confidence=lang.score,
    )

    # Convert the record to JSON
    record_str = json.dumps(record)

    # Return the record and the detected language
    return record_str, lang.name
