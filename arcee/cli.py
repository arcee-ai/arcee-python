from itertools import groupby
from pathlib import Path
from typing import List, Optional

import typer
from click import ClickException as ArceeException
from rich.console import Console
from rich.table import Table
from typing_extensions import Annotated

import arcee.api
from arcee import DALM
from arcee.cli_handler import UploadHandler, WeightsDownloadHandler

console = Console()

cli = typer.Typer()
"""Arcee CLI"""


# FIXME: train_dalm seems to no longer exist...
# @cli.command()
# def train(
#     name: Annotated[str, typer.Argument(help="Name of the model")],
#     context: Annotated[Optional[str], typer.Option(help="Name of the context")] = None,
#     instructions: Annotated[Optional[str], typer.Option(help="Instructions for the model")] = None,
#     generator: Annotated[str, typer.Option(help="Generator type")] = "Command",
# ) -> None:
#     # name: str, context: Optional[str] = None, instructions: Optional[str] = None, generator: str = "Command"
#     """Train a model

#     Args:
#         name (str): Name of the model
#         context (str): Name of the context
#         instructions (str): Instructions for the model
#         generator (str): Generator type. Defaults to "Command".
#     """

#     try:
#         train_dalm(name, context, instructions, generator)
#         typer.secho(f"✅ Model {name} set for training.")
#     except Exception as e:
#         raise ArceeException(
#             message=f"Error training model: {e}",
#         ) from e


@cli.command()
def generate(
    name: Annotated[str, typer.Argument(help="Model name")],
    query: Annotated[str, typer.Option(help="Query string")],
    size: Annotated[int, typer.Option(help="Size of the response")] = 3,
) -> None:
    """Generate from model

    Args:
        name (str): Name of the model
        query (str): Query string
        size (int): Size of the response. Defaults to 3.

    """

    try:
        dalm = DALM(name=name)
        resp = dalm.generate(query=query, size=size)
        typer.secho(resp)
    except Exception as e:
        raise ArceeException(message=f"Error generating: {e}") from e


@cli.command()
def retrieve(
    name: Annotated[str, typer.Argument(help="Model name")],
    query: Annotated[str, typer.Option(help="Query string")],
    size: Annotated[int, typer.Option(help="Size")] = 3,
) -> None:
    """Retrieve from model

    Args:
        name (str): Name of the model
        query (str): Query string
        size (int): Size of the response. Defaults to 3.

    """
    try:
        dalm = DALM(name=name)
        resp = dalm.retrieve(query=query, size=size)
        typer.secho(resp)
    except Exception as e:
        raise ArceeException(message=f"Error retrieving: {e}") from e


upload = typer.Typer(help="Upload data to Arcee platform")


@upload.command()
def context(
    name: Annotated[str, typer.Argument(help="Name of the context")],
    file: Annotated[
        Optional[List[Path]],
        typer.Option(help="Path to a document", exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = None,
    doc_name: Annotated[
        str,
        typer.Option(help="Column/key representing the doc name. Used if file is jsonl or csv", exists=True),
    ] = "name",
    doc_text: Annotated[
        str,
        typer.Option(help="Column/key representing the doc text. Used if file is jsonl or csv", exists=True),
    ] = "text",
    directory: Annotated[
        Optional[List[Path]],
        typer.Option(
            help="Path to a directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
    chunk_size: Annotated[
        int, typer.Option(help="Specify the chunk size in megabytes (MB) to limit memory usage during file uploads.")
    ] = 512,
) -> None:
    """Upload document(s) to context. If a directory is provided, all valid files in the directory will be uploaded.
    At least one of file or directory must be provided.

    If you are using CSV or jsonl file(s), every key/column in your dataset that isn't that of `doc_name` and `doc_text`
    will be uploaded as extra metadata fields with your doc. These can be used for filtering on generation and retrieval

    Args:
        name (str): Name of the context
        file (Path): Path to the file.
        directory (Path): Path to the directory.
        chunk_size (int): The chunk size in megabytes (MB) to limit memory usage during file uploads.
        doc_name (str): The name of the column/key representing the doc name. Used for csv/jsonl
        doc_text (str): The name of the column/key representing the doc text/content. Used for csv/jsonl
    """
    if not file and not directory:
        raise typer.BadParameter("At least one file or directory must be provided")

    if file is None:
        file = []

    if directory is None:
        directory = []

    file.extend(directory)

    try:
        resp = UploadHandler.handle_doc_upload(name, file, chunk_size, doc_name, doc_text)
        typer.secho(resp)
    except Exception as e:
        raise ArceeException(message=f"Error uploading document(s): {e}") from e


@upload.command(hidden=True)  # TODO: - remove hidden=True when vocabulary upload is implemented
def vocabulary(
    name: Annotated[str, typer.Argument(help="Name of the context")],
    file: Annotated[
        Optional[List[Path]],
        typer.Option(help="Path to a document", exists=True, file_okay=True, dir_okay=False, readable=True),
    ] = None,
    directory: Annotated[
        Optional[List[Path]],
        typer.Option(
            help="Path to a directory",
            exists=True,
            file_okay=False,
            dir_okay=True,
        ),
    ] = None,
) -> None:
    """Upload a vocabulary file

    Args:
        name (str): Name of the vocabulary
        file (Path): Path to the file
    """
    if not file and not directory:
        raise typer.BadParameter("Atleast one file or directory must be provided")

    if directory is None:
        directory = []

    if file is None:
        file = []

    file.extend(directory)

    docs = []
    for f in file:
        docs.append({"doc_name": f.name, "doc_text": f.read_text()})

    # TODO: upload_vocabulary
    # Uploadhandler.handle_vocabulary_upload(name, data)


cli.add_typer(upload, name="upload")


cpt = typer.Typer(help="Manage CPT")


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


cli.add_typer(cpt, name="cpt")


sft = typer.Typer(help="Manage SFT")


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


cli.add_typer(sft, name="sft")


retriever = typer.Typer(help="Manage Retrievers")


@retriever.command(name="download")
def download_retriever_weights(
    name: Annotated[
        str,
        typer.Option(
            help="Name of the retriever model to download weights for", prompt="Enter the name of the retriever model"
        ),
    ],
    out: Annotated[
        Optional[Path],
        typer.Option(help="Path to download file to", file_okay=True, dir_okay=False, readable=True),
    ] = None,
) -> None:
    """Download Retriever weights"""
    WeightsDownloadHandler.handle_weights_download("retriever", name, out)


cli.add_typer(retriever, name="retriever")

merging = typer.Typer(help="Manage Merging")


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


cli.add_typer(merging, name="merging")


@cli.command()
def org() -> None:
    """Prints the current org"""
    try:
        result = arcee.api.get_current_org()
        console.print(f"Current org: {result}")
    except Exception as e:
        console.print_exception()
        raise ArceeException(message=f"Error getting current org: {e}") from e


if __name__ == "__main__":
    cli()
