from typing import Any, Literal

import requests

from arcee import config
from arcee.api_handler import make_request, retry_call
from arcee.schemas.routes import Route


def check_model_status(name: str) -> dict[str, str]:
    route = Route.train_model_status.value.format(id_or_name=name)
    return make_request("get", route)


class DALM:
    def __init__(self, name: str) -> None:
        self.name = name

        retriever_api_response = check_model_status(name)

        self.model_id = retriever_api_response["id"]
        self.status = retriever_api_response["status"]

        if self.status != "training_complete":
            raise Exception("DALM model is not ready. Please wait for training to complete.")

        # if ever separate retriever services froma arcee
        # self.retriever_url = retriever_api_response["retriever_url"]
        self.generate_url = config.ARCEE_GENERATION_URL
        self.retriever_url = config.ARCEE_RETRIEVAL_URL

    @retry_call(wait_sec=0.5)
    def invoke(self, invocation_type: Literal["retrieve", "generate"], query: str, size: int) -> dict[str, Any]:
        url = self.retriever_url if invocation_type == "retrieve" else self.generate_url
        payload = {"model_id": self.model_id, "query": query, "size": size}
        headers = {"Authorization": f"Bearer {config.ARCEE_API_KEY}"}

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to {invocation_type}. Response: {response.text}")
        return response.json()

    def retrieve(self, query: str, size: int = 3) -> dict:
        """Retrieve {size} contexts with your retriever for the given query"""
        return self.invoke("retrieve", query, size)

    def generate(self, query: str, size: int = 3) -> dict:
        """Generate a response using {size} contexts with your generator for the given query"""
        return self.invoke("generate", query, size)
