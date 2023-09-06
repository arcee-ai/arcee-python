__version__ = "0.0.8"
import os
import requests
import json
from arcee.dalm import DALM
from arcee.dalm import check_model_status
from arcee.config import ARCEE_API_KEY, ARCEE_API_URL, ARCEE_APP_URL, ARCEE_API_VERSION
import time

if ARCEE_API_KEY is None:
    raise Exception(f"ARCEE_API_KEY must be in the environment. You can retrieve your API key from {ARCEE_APP_URL}")

def upload_doc(context, doc_name, doc_text):
    """
    Upload a document to a context

    Args:
        context (str): The name of the context to upload to
        doc_name (str): The name of the document
        doc_text (str): The text of the document
    """
    doc = {
        "name": doc_name,
        "document": doc_text
    }

    headers = {
        "X-Token": f"{ARCEE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "context_name": context,
        "documents": [doc]
    }

    response = requests.post(f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/upload-context", headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        raise Exception(f"Failed to upload example. Response: {response.text}")

    return response.json()

def upload_docs(context, docs):
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

        doc_list.append({
            "name": doc["doc_name"],
            "document": doc["doc_text"]
        })

    headers = {
        "X-Token": f"{ARCEE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "context_name": context,
        "documents": doc_list
    }

    response = requests.post(f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/upload-context", headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        raise Exception(f"Failed to upload example. Response: {response.text}")

    return response.json()


def train_dalm(name, context=None, instructions=None, generator="Command"):

    endpoint = f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/train-model"
    data_to_send = {
        "name": name,
        "context": context,
        "instructions": instructions,
        "generator": generator
    }

    headers = {
        "X-Token": f"{ARCEE_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint, data=json.dumps(data_to_send), headers=headers)

    if response.status_code != 201:
        raise Exception(f"Failed to train model. Response: {response.text}")
    else:
        print("DALM model training started - view model status at {ARCEE_APP_URL}, then arcee.get_model(" + name + ")")

def get_dalm(name):
    return DALM(name)