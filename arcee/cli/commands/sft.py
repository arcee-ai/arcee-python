from pathlib import Path
from typing import Optional

import typer
from typing_extensions import Annotated

from arcee.cli.handlers.weights import WeightsDownloadHandler
from arcee.cli.typer import ArceeTyper

sft = ArceeTyper(
    help="Manage SFT", epilog="For more information on SFT, see https://docs.arcee.ai/aligning/should-i-align-my-model"
)


@sft.command(name="download")
def download_sft_weights(
    name: Annotated[
        str,
        typer.Option(help="Name of the SFT model to download weights for", prompt="Enter the name of the SFT model"),
    ],
    out: Annotated[
        Optional[Path],
        typer.Option(help="Path to download file to", file_okay=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Download SFT weights"""
    WeightsDownloadHandler.handle_weights_download("alignment", name, out)
