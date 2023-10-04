from importlib.util import find_spec
from pathlib import Path
from typing import Dict, List

import typer
from click import ClickException as ArceeException
from rich.progress import Progress, SpinnerColumn, TextColumn

from arcee import upload_docs
from arcee.schemas.doc import Doc

if not find_spec("pandas"):
    raise ModuleNotFoundError("Cannot find pandas. Please run `pip install 'arcee-py[cli]'` for cli support")

import pandas as pd


class UploadHandler:
    """Upload data to Arcee platform"""

    valid_context_file_extensions = {".txt", ".jsonl", ".csv"}

    one_kb = 1024
    one_mb = 1024 * one_kb
    one_gb = 1024 * one_mb

    @classmethod
    def _validator(cls, paths: List[Path]) -> List[Path]:
        """Validates file paths.

        Validations:
            - path is a file
            - path has a valid extension `.txt` or `.jsonl`

        Args:
            paths List[Path]: list of paths to files.

        Returns:
            List[Path]: Validated unique paths

        Raises:
            typer.BadParameter: If any path is not a file or has an invalid extension
        """
        for path in paths:
            if not path.is_file() or path.suffix not in cls.valid_context_file_extensions:
                raise typer.BadParameter(
                    f"{path} is not a file or has an invalid extension;"
                    f"\nAllowed {' '.join(cls.valid_context_file_extensions)}"
                )

        return paths

    @classmethod
    def _handle_paths(cls, paths: List[Path]) -> List[Path]:
        """Process paths and spread them into constituent files if path is a directory.

        Args:
            paths List[Path]: list of paths.

        Returns:
            List[Path]: unique list of paths.
        """
        all_paths: List[Path] = []
        for path in paths:
            if not path.is_dir():
                all_paths.append(path)  # append any path that's not a directory
            else:
                all_paths.extend(
                    filter(lambda x: not x.is_dir(), path.iterdir())
                )  # append all non directory paths of a directory

        return list(set(all_paths))

    @classmethod
    def _get_docs(cls, file: Path, doc_name: str, doc_text: str) -> List[Doc]:
        if file.suffix == ".txt":
            return [Doc(doc_name=file.name, doc_text=file.read_text())]
        if file.suffix == ".jsonl":
            df = pd.read_json(file.name, lines=True)
        elif file.suffix == ".csv":
            df = pd.read_csv(file.name)
        else:
            raise ValueError(f"File type not valid. Must be one of {cls.valid_context_file_extensions}")
        if doc_name not in df.columns:
            raise ValueError(
                f"{doc_name} not found in data column/key. Rename column/key or use "
                f"--doc-name in comment to specify your own"
            )
        if doc_text not in df.columns:
            raise ValueError(
                f"{doc_text} not found in data column/key. Rename column/key or use "
                f"--doc-text in comment to specify your own"
            )
        return [Doc(doc_name=row.pop(doc_name), doc_text=row.pop(doc_text), meta=dict(row)) for _, row in df.iterrows()]

    @classmethod
    def _handle_upload(
        cls, name: str, files: List[Path], max_chunk_size: int, doc_name: str, doc_text: str
    ) -> Dict[str, str]:
        """Upload document file(s) to context
        Args:
            name str: Name of the context
            files List[Path]: tuple of paths to valid file(s).
            max_chunk_size int: Maximum memory, in bytes to use for uploading
        """
        docs: List[Dict[str, str]] = []
        chunk: int = 0
        for file in files:
            if chunk + file.stat().st_size >= max_chunk_size:
                if len(docs) == 0:
                    raise ArceeException(
                        message=f"Memory Limit Exceeded. "
                        f"When uploading {file.name} ({file.stat().st_size/cls.one_mb} MB). "
                        "Try increasing chunk size."
                    )
                upload_docs(context=name, docs=docs)
                chunk = 0
                docs.clear()
            chunk += file.stat().st_size
            file_docs = cls._get_docs(file, doc_name, doc_text)
            file_docs_json = [doc.dict() for doc in file_docs]
            docs.extend(file_docs_json)

        return upload_docs(context=name, docs=docs)

    @classmethod
    def handle_doc_upload(
        cls, name: str, paths: List[Path], chunk_size: int, doc_name: str, doc_text: str
    ) -> Dict[str, str]:
        """Handle document upload from valid paths to files and directories

        Args:
            name str: Name of the context.
            paths List[Path]: tuple of paths to files or directories.
            chunk_size int: Maximum memory in megabytes (MB) to use for uploading
        """
        paths_validator = cls._validator
        paths_handler = cls._handle_paths
        doc_uploader = cls._handle_upload
        ONE_MB = cls.one_mb

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            # process paths
            processing = progress.add_task(description=f"Processing {len(paths)} path(s)...", total=len(paths))
            paths = paths_handler(paths)
            progress.update(processing, description=f"✅ Listed {len(paths)} document path(s)")

            # validate paths
            validating = progress.add_task(description=f"Validating {len(paths)} path(s)...", total=len(paths))
            files = paths_validator(paths)
            progress.update(validating, description=f"✅ Validated {len(paths)} files(s)")

            # upload documents
            uploading = progress.add_task(description=f"Uploading {len(paths)} document(s)...", total=len(files))
            resp = doc_uploader(
                name=name, files=files, max_chunk_size=chunk_size * ONE_MB, doc_name=doc_name, doc_text=doc_text
            )
            progress.update(uploading, description=f"✅ Uploaded {len(paths)} document(s) to context {name}")
            return resp
