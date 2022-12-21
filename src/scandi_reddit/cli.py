"""Command line interface for creating a Scandinavian Reddit dataset."""

from typing import Optional

import click

from .build import build_reddit_dataset


@click.command()
@click.option(
    "--overwrite/--no-overwrite",
    "-o",
    default=False,
    help="Overwrite existing files.",
)
@click.option(
    "--n-jobs",
    "-j",
    default=-2,
    help="The number of jobs to run in parallel.",
)
@click.option(
    "--starting-year",
    "-y",
    default=2005,
    help="The year to start downloading from. Defaults to 2005.",
)
@click.option(
    "--starting-month",
    "-m",
    default=1,
    help="The month to start downloading from. Defaults to 1.",
)
@click.option(
    "--skip-download/--no-skip-download",
    default=False,
    help="Whether to skip downloading the files.",
)
@click.option(
    "--hub-repo-id",
    default=None,
    help="The ID of the Hugging Face Hub repository to upload the dataset to.",
)
def main(
    overwrite: bool,
    n_jobs: int,
    starting_year: int,
    starting_month: int,
    skip_download: bool,
    hub_repo_id: Optional[str],
) -> None:
    """Build a Scandinavian Reddit dataset."""
    build_reddit_dataset(
        overwrite=overwrite,
        n_jobs=n_jobs,
        starting_year=starting_year,
        starting_month=starting_month,
        skip_download=skip_download,
        hub_repo_id=hub_repo_id,
    )


if __name__ == "__main__":
    main()
