from typing import Any

import typer


class ArceeTyper(typer.Typer):
    """
    Custom Typer class for Arcee CLI

    Use this instead of `typer.Typer` to set default values for the CLI
    and consistency across all commands.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        # UX
        kwargs.setdefault("no_args_is_help", True)

        # Security
        kwargs.setdefault("pretty_exceptions_show_locals", False)

        # Formatting
        kwargs.setdefault("rich_markup_mode", "rich")

        super().__init__(*args, **kwargs)
