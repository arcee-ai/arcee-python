from itertools import groupby
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

import arcee.api
from arcee.cli.errors import ArceeException
from arcee.cli.handlers.weights import WeightsDownloadHandler
from arcee.cli.typer import ArceeTyper

console = Console()

cpt = ArceeTyper(
    rich_markup_mode="rich",
    no_args_is_help=True,
    help="""
        Manage CPT models and weights on the Arcee platform
    """,
    epilog="For more information on CPT, see https://docs.arcee.ai/pretraining/should-i-pretrain",
)


@cpt.command(name="list")
def list_cpts() -> None:
    """List all CPTs"""
    try:
        result = arcee.api.list_pretrainings()

        table = Table("Name", "Status", "Base Generator", "Last Updated", title="List CPTs")

        key_func = lambda x: x["processing_state"]  # noqa: E731
        grouped_data = {key: list(group) for key, group in groupby(result, key_func)}

        captions = []

        for key, group in grouped_data.items():
            if key == "failed":
                captions.append(typer.style(f"Failed: {len(group)}", fg="red"))
            elif key == "completed":
                captions.append(typer.style(f"Completed: {len(group)}", fg="green"))
            elif key == "processing":
                captions.append(typer.style(f"Processing: {len(group)}", fg="yellow"))
            elif key == "pending":
                captions.append(typer.style(f"Pending: {len(group)}", fg="blue"))

            table.add_section()

            for cpt in list(group):
                table.add_row(
                    cpt["name"],
                    cpt["status"],
                    cpt["base_generator"],
                    cpt.get("updated_at") or cpt.get("created_at", "-"),
                )

        if len(captions) > 0:
            table.caption = " | ".join(captions)
        console.print(table)
    except Exception as e:
        raise ArceeException(message=f"Error listing CPTs: {e}") from e


@cpt.command(name="download")
def download_cpt_weights(
    name: Annotated[
        str,
        typer.Option(help="Name of the CPT model to download weights for", prompt="Enter the name of the CPT model"),
    ],
    out: Annotated[
        Optional[Path],
        typer.Option(help="Path to download file to", file_okay=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Download CPT weights"""
    WeightsDownloadHandler.handle_weights_download("pretraining", name, out)
