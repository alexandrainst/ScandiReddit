"""Command line interface for creating a Scandinavian Reddit dataset."""

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
def main(
    overwrite: bool,
    n_jobs: int,
    starting_year: int,
    starting_month: int,
    skip_download: bool,
) -> None:
    """Build a Scandinavian Reddit dataset."""
    build_reddit_dataset(
        overwrite=overwrite,
        n_jobs=n_jobs,
        starting_year=starting_year,
        starting_month=starting_month,
        skip_download=skip_download,
    )


if __name__ == "__main__":
    main()
