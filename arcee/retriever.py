import os
import requests
import json
from arcee.config import ARCEE_API_URL, ARCEE_QUERY_URL, ARCEE_API_KEY

def check_retriever_status(context):

    endpoint = f"{ARCEE_API_URL}/get-retriever-status"
    # Data you wish to send
    data_to_send = {
        "context_name": context
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ARCEE_API_KEY}"
    }

    response = requests.post(endpoint, data=json.dumps(data_to_send), headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to check retriever status. Response: {response.text}")
    else:
        return response.json()

class Retriever:
    def __init__(self, context):

        self.context = context

        retriever_api_response = check_retriever_status(context)

        self.context_id = retriever_api_response["context_id"]
        self.status = retriever_api_response["status"]

        if self.status != "training_complete":
            raise Exception("Retriever is not ready. Please wait for training to complete.")

        #if ever separate retriever services froma arcee
        #self.retriever_url = retriever_api_response["retriever_url"]
        self.retriever_url = ARCEE_QUERY_URL

    def retrieve(self, query, size=3):
        """Retrieve a  from a given URL"""
        payload = {
            "context_id": self.context_id,
            "query": query
        }

        headers = {
            "Authorization": f"Bearer {os.environ['ARCEE_API_KEY']}"
        }

        response = requests.post(self.retriever_url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve. Response: {response.text}")

        return response.json()
        
        