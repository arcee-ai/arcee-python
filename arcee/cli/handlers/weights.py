from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import DownloadColumn, Progress, TimeElapsedColumn, TransferSpeedColumn

from arcee.api import download_weights, model_weight_types
from arcee.cli.errors import ArceeException

console = Console()


class WeightsDownloadHandler:
    """Download weights from Arcee platform"""

    @classmethod
    def handle_weights_download(cls, kind: model_weight_types, id_or_name: str, path: Optional[Path] = None) -> None:
        """Download weights from Arcee platform

        Args:
            kind model_weight_types: Type of model weights.
            id_or_name str: Name or ID of the model.
            path Path: Path to save the weights.
        """
        try:
            out = path or Path.cwd() / f"{id_or_name}.tar.gz"
            console.print(f"Downloading {kind} model weights for {id_or_name} to {out}")

            with open(out, "wb") as f:
                with download_weights(kind, id_or_name) as response:
                    response.raise_for_status()
                    size = int(response.headers.get("Content-Length", 0))

                    with Progress(
                        *Progress.get_default_columns(),
                        DownloadColumn(),
                        TimeElapsedColumn(),
                        TransferSpeedColumn(),
                        transient=True,
                    ) as progress:
                        task = progress.add_task(
                            f"[blue]Downloading {id_or_name} weights...", total=size if size > 0 else None
                        )

                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
                            progress.update(task, advance=len(chunk))

                        console.print(f"Downloaded {out} in {progress.get_time()} seconds")
        except Exception as e:
            console.print_exception()
            raise ArceeException(message=f"Error downloading {kind} weights: {e}") from e
