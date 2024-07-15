from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from arcee.cli.handlers.weights import WeightsDownloadHandler
from arcee.cli.typer import ArceeTyper

merging = ArceeTyper(
    help="Manage Merging", epilog="For more information on merging, see https://docs.arcee.ai/merging/should-i-merge"
)


@merging.command(name="download")
def download_merging_weights(
    name: Annotated[
        str,
        typer.Option(
            help="Name of the merging model to download weights for", prompt="Enter the name of the merging model"
        ),
    ],
    out: Annotated[
        Optional[Path],
        typer.Option(help="Path to download file to", file_okay=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Download Merging weights"""
    WeightsDownloadHandler.handle_weights_download("merging", name, out)
