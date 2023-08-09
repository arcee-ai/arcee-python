__version__ = "0.0.6"
import os
import requests
import json


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

def upload_doc(context, name, document_text, summary=None):
    """
    Upload a document to a context

    Args:
        context (str): The name of the context to upload to
        name (str): The name of the document
        document_text (str): The text of the document
        summary (str, optional): The summary of the document. Defaults to None. Summary will be the first 500 characters of the document if not provided.
    """
    ARCEE_API_KEY = os.getenv("ARCEE_API_KEY")
    if ARCEE_API_KEY is None:
        raise Exception("ARCEE_API_KEY must be in the environment")

    doc = {
        "name": name,
        "summary": summary if summary is not None else document_text[:500],
        "document": document_text
    }

    headers = {
        "Authorization": f"Bearer {ARCEE_API_KEY}",
        "Content-Type": "application/json"
    }
    data = {
        "context_name": context,
        "document": doc
    }

    response = requests.post("http://127.0.0.1:9001/v1/upload-context", headers=headers, data=json.dumps(data))

    if response.status_code != 200:
        raise Exception(f"Failed to upload example. Response: {response.text}")

    return response.json()