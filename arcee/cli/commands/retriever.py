from pathlib import Path
from typing import List, Optional

import typer
from typing_extensions import Annotated

from arcee.cli.errors import ArceeException
from arcee.cli.handlers.upload import UploadHandler
from arcee.cli.handlers.weights import WeightsDownloadHandler
from arcee.cli.typer import ArceeTyper
from arcee.dalm import DALM

retriever = ArceeTyper(help="Manage Retrievers")


@retriever.command()
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


@retriever.command()
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


@retriever.command(name="upload-context", short_help="Upload document(s) to context")
def upload_context(
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
