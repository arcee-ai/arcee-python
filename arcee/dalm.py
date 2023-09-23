import requests

from arcee.api_handler import make_request
from arcee.config import ARCEE_API_KEY, ARCEE_GENERATION_URL, ARCEE_RETRIEVAL_URL
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
        self.generate_url = ARCEE_GENERATION_URL
        self.retriever_url = ARCEE_RETRIEVAL_URL

    def retrieve(self, query: str, size: int = 3) -> dict:
        """Retrieve a  from a given URL"""
        payload = {"model_id": self.model_id, "query": query, "size": size}

        headers = {"Authorization": f"Bearer {ARCEE_API_KEY}"}

        response = requests.post(self.retriever_url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve. Response: {response.text}")

        return response.json()

    def generate(self, query: str, size: int = 3) -> dict:
        payload = {"model_id": self.model_id, "query": query, "size": size}

        headers = {"Authorization": f"Bearer {ARCEE_API_KEY}"}

        response = requests.post(self.generate_url, json=payload, headers=headers)

        if response.status_code != 200:
            raise Exception(f"Failed to generate. Response: {response.text}")

        return response.json()
