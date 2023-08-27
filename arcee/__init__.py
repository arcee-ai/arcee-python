__version__ = "0.0.6"
import os
import requests
import json
from arcee.retriever import Retriever
from arcee.retriever import check_retriever_status
from arcee.config import ARCEE_API_KEY, ARCEE_API_URL, ARCEE_APP_URL, ARCEE_API_VERSION
import time
# import arcee
# for doc in docs:
    # arcee.upload_doc(context="pubmed", name=doc["name"], document=doc["document"])

# retriever = arcee.train_retriever(context="pubmed", target_generator="GPT-4")
# Get out of GPT-4 as soon as possible, Bedrock claude, cohere, etc. - llama+arcee domain on sagemaker

# retriever.retrieve("what are the components of Scopolamine?")
##>>Context 1: Scopolamine study1
##>>Context 2: Scopolamine study2
##>>Context 3: Scopolamine study3
##>>Retrieval Time: 25ms

# retriever.retrieve_and_generate("what are the components of Scopolamine?", generator="GPT-4")
##>> ADD GENERATOR FOR DEMO

if ARCEE_API_KEY is None:
    raise Exception(f"ARCEE_API_KEY must be in the environment. You can retrieve your API key from {ARCEE_APP_URL}")

def upload_doc(context, name, document_text, summary=None):
    """
    Upload a document to a context

    Args:
        context (str): The name of the context to upload to
        name (str): The name of the document
        document_text (str): The text of the document
        summary (str, optional): The summary of the document. Defaults to None. Summary will be the first 500 characters of the document if not provided.
    """
    doc = {
        "name": name,
        "summary": summary if summary is not None else document_text[:500],
        "document": document_text
    }

    headers = {
        "X-Token": f"{ARCEE_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "context_name": context,
        "document": doc
    }

    #response = requests.post("http://localhost:8000/v1/upload-context", headers=headers, data=json.dumps(data))
    response = requests.post(f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/upload-context", headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        raise Exception(f"Failed to upload example. Response: {response.text}")

    return response.json()

def train_retriever(context):

    endpoint = f"{ARCEE_API_URL}/{ARCEE_API_VERSION}/train-retriever"
    data_to_send = {
        "context_name": context
    }

    headers = {
        "X-Token": f"{ARCEE_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(endpoint, data=json.dumps(data_to_send), headers=headers)

    if response.status_code != 201:
        raise Exception(f"Failed to train retriever. Response: {response.text}")

    print(f"Retriever training started - view retriever status at {ARCEE_APP_URL}")

    current_status = "machine_starting"

    while current_status != "training_complete":
        current_status = check_retriever_status(context)["status"]
        print("Current status: ", current_status)

        time.sleep(5)

        if current_status == "training_complete":
            print("Retriever training complete")
            break

    return Retriever(context)

def get_retriever(context):
    return Retriever(context)