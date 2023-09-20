from pathlib import Path

import typer
from rich.progress import Progress, SpinnerColumn, TextColumn

from arcee import upload_doc, upload_docs


class UploadHandler:
    """Upload data to Arcee platform"""

    valid_context_file_extensions = set([".txt", ".jsonl"])

    @classmethod
    def _validator(cls, paths: list[Path]) -> list[Path]:
        """Validates file paths.

        Validations:
            - path is a file
            - path has a valid extension `.txt` or `.jsonl`

        Args:
            paths list[Path]: list of paths to files.

        Returns:
            list[Path]: Validated unique paths

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
    def _handle_paths(cls, paths: list[Path]) -> list[Path]:
        """Process paths and spread them into constituent files if path is a directory.

        Args:
            paths list[Path]: list of paths.

        Returns:
            list[Path]: unique list of paths.
        """
        all_paths: list[Path] = []
        for path in paths:
            if not path.is_dir():
                all_paths.append(path)  # append any path that's not a directory
            else:
                all_paths.extend(
                    filter(lambda x: not x.is_dir(), path.iterdir())
                )  # append all non directory paths of a directory

        return list(set(all_paths))

    @classmethod
    def _handle_upload(cls, name: str, files: list[Path]) -> dict[str, str]:
        """Upload document file(s) to context
        Args:
            name (str): Name of the context
            files list[Path]: tuple of paths to valid file(s).
        """

        # if only one file is passed, upload it
        if len(files) == 1:
            file = files[0]
            return upload_doc(context=name, doc_name=file.name, doc_text=file.read_text())

        docs: list[dict[str, str]] = []

        for file in files:
            docs.append({"doc_name": file.name, "doc_text": file.read_text()})

        return upload_docs(context=name, docs=docs)

    @classmethod
    def handle_doc_upload(cls, name: str, paths: list[Path]) -> dict[str, str]:
        """Handle document upload from valid paths to files and directories

        Args:
            name (str): Name of the context
            paths list[Path]: tuple of paths to files or directories.
        """
        paths_validator = cls._validator
        paths_handler = cls._handle_paths
        doc_uploader = cls._handle_upload

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=False,
        ) as progress:
            # process paths
            processing = progress.add_task(description=f"Processing {len(paths)} path(s)...", total=len(paths))
            paths = paths_handler(paths)
            progress.update(processing, description=f"✅ Listed {len(paths)} document(s)")

            # validate paths
            validating = progress.add_task(description=f"Validating {len(paths)} path(s)...", total=len(paths))
            files = paths_validator(paths)
            progress.update(validating, description=f"✅ Validated {len(paths)} path(s)")

            # upload documents
            uploading = progress.add_task(description=f"Uploading {len(paths)} document(s)...", total=len(files))
            resp = doc_uploader(name, files)
            progress.update(uploading, description=f"✅ Uploaded {len(paths)} document(s) to context {name}")
            return resp
