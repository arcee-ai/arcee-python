import os
import requests

class Retriever:
    def __init__(self, context):

        if "ARCEE_API_KEY" not in os.environ:
            raise Exception("ARCEE_API_KEY must be in the environment to initialize a Retriever")

        self.context = context

        #api request here eventually for separate retrieval services
        self.context_id = "f086132a-b205-40c1-9ad3-f5a5d63375b1"
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
        
        