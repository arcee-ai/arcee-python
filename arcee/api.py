from typing import Optional

from arcee.api_handler import make_request
from arcee.config import ARCEE_APP_URL
from arcee.dalm import DALM, check_model_status
from arcee.schemas.routes import Route


def upload_doc(context: str, doc_name: str, doc_text: str) -> dict[str, str]:
    """
    Upload a document to a context

    Args:
        context (str): The name of the context to upload to
        doc_name (str): The name of the document
        doc_text (str): The text of the document
    """
    doc = {"name": doc_name, "document": doc_text}
    data = {"context_name": context, "documents": [doc]}
    return make_request("post", Route.contexts, data)


def upload_docs(context: str, docs: list[dict[str, str]]) -> dict[str, str]:
    """
    Upload a list of documents to a context

    Args:
        context (str): The name of the context to upload to
        docs (list): A list of dictionaries with keys "doc_name" and "doc_text"
    """
    doc_list = []
    for doc in docs:
        if "doc_name" not in doc.keys() or "doc_text" not in doc.keys():
            raise Exception("Each document must have a doc_name and doc_text key")

        doc_list.append({"name": doc["doc_name"], "document": doc["doc_text"]})

    data = {"context_name": context, "documents": doc_list}
    return make_request("post", Route.contexts, data)


def train_dalm(
    name: str, context: Optional[str] = None, instructions: Optional[str] = None, generator: str = "Command"
) -> None:
    data = {"name": name, "context": context, "instructions": instructions, "generator": generator}
    make_request("post", Route.train_model, data)
    # TODO: Add org in url
    status_url = f"{ARCEE_APP_URL}/models/{name}/training"
    print(
        f"DALM model training started - view model status at {status_url} and click on your model.\n"
        f'When training is finished, get DALM with arcee.get_dalm("{name}")'
    )


def get_dalm_status(id_or_name: str) -> dict[str, str]:
    """Gets the status of a DALM training job"""
    return check_model_status(id_or_name)


def get_dalm(name: str) -> DALM:
    return DALM(name)
