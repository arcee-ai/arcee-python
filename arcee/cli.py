from pathlib import Path
from typing import List, Optional

import typer
from click import ClickException as ArceeException
from typing_extensions import Annotated

from arcee import DALM, train_dalm
from arcee.cli_handler import UploadHandler

cli = typer.Typer()
"""Arcee CLI"""


@cli.command()
def train(
    name: Annotated[str, typer.Argument(help="Name of the model")],
    context: Annotated[Optional[str], typer.Option(help="Name of the context")] = None,
    instructions: Annotated[Optional[str], typer.Option(help="Instructions for the model")] = None,
    generator: Annotated[str, typer.Option(help="Generator type")] = "Command",
) -> None:
    # name: str, context: Optional[str] = None, instructions: Optional[str] = None, generator: str = "Command"
    """Train a model

    Args:
        name (str): Name of the model
        context (str): Name of the context
        instructions (str): Instructions for the model
        generator (str): Generator type. Defaults to "Command".
    """

    try:
        train_dalm(name, context, instructions, generator)
        typer.secho(f"✅ Model {name} set for training.")
    except Exception as e:
        raise ArceeException(
            message=f"Error training model: {e}",
        ) from e


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

if __name__ == "__main__":
    cli()
