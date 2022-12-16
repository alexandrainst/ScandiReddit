"""Downloads Reddit comments from the Pushshift API."""

import logging
from pathlib import Path
from urllib.error import ContentTooShortError, HTTPError

import wget

# Set up logger
logger = logging.getLogger(__name__)


def download_reddit_file(year: int, month: int) -> Path:
    """Downloads a Reddit dump for a given year and month.

    Args:
        year (int):
            The year to download.
        month (int):
            The month to download.

    Returns:
        Path:
            The path to the downloaded file.
    """
    # Create the file name
    raw_data_dir = Path("data") / "raw"
    path = raw_data_dir / f"RC_{year}-{month:02d}.zst"

    # Download the file if it doesn't exist, skip if the download fails
    while not path.exists():
        try:
            wget.download(
                f"https://files.pushshift.io/reddit/comments/{path.name}",
                out=path.name,
            )

        # If we get a ContentTooShortError, then we remove the file and try again
        except ContentTooShortError:
            path.unlink(missing_ok=True)
            for tmp_file in raw_data_dir.glob("RC_*.tmp"):
                tmp_file.unlink(missing_ok=True)
            logger.debug("Download failed, retrying...")

        # If we get an HTTPError then assume that the file doesn't exist and stop
        # trying
        except HTTPError:
            break

    # Return the path, no matter if the file exists or not
    return path
