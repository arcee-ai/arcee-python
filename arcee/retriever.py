import os
import requests
import json

def check_retriever_status(context):
    # Base API URL and version
    BASE_URL = "http://127.0.0.1:9001"  # replace with your actual API endpoint
    API_VERSION = "v1"  # replace with your actual API version if different

    # Endpoint for train_retriever
    endpoint = f"{BASE_URL}/{API_VERSION}/get-retriever-status"

    # Data you wish to send
    data_to_send = {
        "context_name": context
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {os.environ['ARCEE_API_KEY']}"
    }

    response = requests.post(endpoint, data=json.dumps(data_to_send), headers=headers)

    if response.status_code != 200:
        raise Exception(f"Failed to check retriever status. Response: {response.text}")
    else:
        return response.json()

class Retriever:
    def __init__(self, context):

        if "ARCEE_API_KEY" not in os.environ:
            raise Exception("ARCEE_API_KEY must be in the environment to initialize a Retriever")

        self.context = context

        retriever_api_response = check_retriever_status(context)

        self.context_id = retriever_api_response["context_id"]
        self.status = retriever_api_response["status"]

        if self.status != "training_complete":
            raise Exception("Retriever is not ready. Please wait for training to complete.")
        #self.retriever_url = retriever_api_response["retriever_url"]
        self.retriever_url = "https://mbrbj7kufwdir2uy2lox2osvcy0hmvhn.lambda-url.us-east-2.on.aws/retrieve"

    def retrieve(self, query, size=3):
        """Retrieve a  from a given URL"""
        payload = {
            "context_id": self.context_id,
            "query": query
        }

        response = requests.post(self.retriever_url, json=payload)

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve. Response: {response.text}")

        return response.json()
        
        