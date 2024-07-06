from typing import Optional

import typer
from rich.console import Console
from rich.prompt import Prompt
from typing_extensions import Annotated

import arcee.api
from arcee.cli.commands.cpt import cpt
from arcee.cli.commands.merging import merging
from arcee.cli.commands.retriever import retriever
from arcee.cli.commands.sft import sft
from arcee.cli.errors import ArceeException
from arcee.cli.typer import ArceeTyper
from arcee.config import ARCEE_API_KEY, ARCEE_API_URL, ARCEE_ORG, write_configuration_value

console = Console()

"""Arcee CLI"""
cli = ArceeTyper(
    help=f"""
        Welcome to the Arcee CLI! 🚀

        This CLI provides a convenient way to interact with the Arcee platform.
        The Arcee client is also available as a Python package for programmatic access.
        {ARCEE_API_URL}/docs
    """,
    epilog="For more information, see our documentation at https://docs.arcee.ai",
)

############################
#  Subcommands
############################
cli.add_typer(cpt, name="cpt")
cli.add_typer(merging, name="merging")
cli.add_typer(retriever, name="retriever")
cli.add_typer(sft, name="sft")


############################
# Top-level CLI Commands
############################
@cli.command()
def org() -> None:
    """Prints the current org"""
    try:
        result = arcee.api.get_current_org()
        console.print(f"Current org: {result}")
    except Exception as e:
        console.print_exception()
        raise ArceeException(message=f"Error getting current org: {e}") from e


@cli.command()
def configure(
    org: Annotated[
        Optional[str],
        typer.Option(
            help="Your organization. If not provided, we will use your default organization. "
            + "Defaults to the ARCEE_ORG environment variable."
        ),
    ] = None,
    api_key: Annotated[
        Optional[str], typer.Option(help="Your API key. Defaults to the ARCEE_API_KEY environment variable.")
    ] = None,
    api_url: Annotated[
        Optional[str],
        typer.Option(help="The URL of the Arcee API. Defaults to ARCEE_API_URL, or https://app.arcee.ai/api."),
    ] = None,
) -> None:
    """Write a configuration file for the Arcee SDK and CLI"""

    if ARCEE_ORG:
        console.print(f"Current org: {ARCEE_ORG}")
    if org:
        console.print(f"Setting org to {org}")
        write_configuration_value("ARCEE_ORG", org)

    if ARCEE_API_URL:
        console.print(f"Current API URL: {ARCEE_API_URL}")
    if api_url:
        console.print(f"Setting API URL to {api_url}")
        write_configuration_value("ARCEE_API_URL", api_url)

    console.print(f"API key: {"in" if ARCEE_API_KEY else "not in"} config (file or env)")

    if api_key:
        console.print("Setting API key")
        write_configuration_value("ARCEE_API_KEY", api_key)

    if not ARCEE_API_KEY and not api_key:
        resp = Prompt.ask(
            # password=True,
            prompt="""
Enter your Arcee API key :lock:
Hit enter to leave it as is.
See https://docs.arcee.ai/getting-arcee-api-key/getting-arcee-api-key for more details.
You can also pass this at runtime with the ARCEE_API_KEY environment variable.
""",
        )
        if resp:
            console.print("Setting API key")
            write_configuration_value("ARCEE_API_KEY", resp)


# Enter the CLI
if __name__ == "__main__":
    cli()
