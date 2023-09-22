import json
from typing import Optional

import requests

from arcee.config import ARCEE_API_KEY, ARCEE_API_URL, ARCEE_API_VERSION, ARCEE_APP_URL
from arcee.dalm import DALM, check_model_status


def upload_doc(context: str, doc_name: str, doc_text: str) -> dict[str, str]:
    """
    Upload a document to a context

    Args:
        context (str): The name of the context to upload to
        doc_name (str): The name of the document
        doc_text (str): The text of the document
    """
    doc = {"name": doc_name, "document": doc_text}

    headers = {"X-Token": f"{ARCEE_API_KEY}", "Content-Type": "application/json"}

    data = {"context_name": context, "documents": [doc]}

    response = requests.post(
        f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/upload-context", headers=headers, data=json.dumps(data)
    )

    if response.status_code != 200:
        raise Exception(f"Failed to upload example. Response: {response.text}")

    return response.json()


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

    headers = {"X-Token": f"{ARCEE_API_KEY}", "Content-Type": "application/json"}

    data = {"context_name": context, "documents": doc_list}

    response = requests.post(
        f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/upload-context", headers=headers, data=json.dumps(data)
    )

    if response.status_code != 200:
        raise Exception(f"Failed to upload example. Response: {response.text}")

    return response.json()


def train_dalm(
    name: str, context: Optional[str] = None, instructions: Optional[str] = None, generator: str = "Command"
) -> None:
    endpoint = f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/train-model"
    data_to_send = {"name": name, "context": context, "instructions": instructions, "generator": generator}

    headers = {"X-Token": f"{ARCEE_API_KEY}", "Content-Type": "application/json"}

    response = requests.post(endpoint, data=json.dumps(data_to_send), headers=headers)

    if response.status_code != 201:
        raise Exception(f"Failed to train model. Response: {response.text}")
    status_url = f"{ARCEE_APP_URL}/arcee/models/{name}/training"
    print(
        f"DALM model training started - view model status at {status_url} or with `arcee.get_dalm_status({name}).\n"
        f"Then, get your DALM with arcee.get_dalm({name})"
    )


def get_dalm_status(id_or_name: str) -> dict[str, str]:
    """Gets the status of a DALM training job"""
    return check_model_status(id_or_name)


def get_dalm(name: str) -> DALM:
    return DALM(name)
