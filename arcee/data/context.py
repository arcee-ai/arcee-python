from datasets import load_dataset
import os
import requests
import json

#Context is a list of documents for your model to observe at inference time

class Context:
    """
    An instruction set is a collection of instructions and responses that is used to instruction finetune an LM.
    """
    def __init__(
        self,
        data_file
    ):
        if not data_file.endswith(".csv"):
            raise (Exception("Datafile must be a .csv file"))

        self.dataset = load_dataset("csv", data_files=data_file, split="train")
        dataset_fields = list(self.dataset.features.keys())
        self.dataset_fields = dataset_fields

        if "name" not in dataset_fields:
            raise (Exception("Datafile must contain a column named 'name'"))

        if "document" not in dataset_fields:
            raise (Exception("Datafile must contain a column named 'completion'"))

        if "summary" not in dataset_fields:
            #if no summary provided, index the first 500 characters of the document
            summary_column = [
                a[:500] for a in self.dataset["document"]
            ]
            self.dataset = self.dataset.add_column("summary", summary_column)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This function formats an entry as it will appear during finetuning
        """
        return f"{self.dataset[idx]['name']} {self.dataset[idx]['summary']}"

    def upload(self, context_name):
        ARCEE_API_KEY = os.getenv("ARCEE_API_KEY")
        if ARCEE_API_KEY is None:
            raise Exception("ARCEE_API_KEY must be in the environment")

        for i in range(len(self)):
            doc = {
                "name": self.dataset[i]["name"],
                "summary": self.dataset[i]["summary"],
                "document": self.dataset[i]["document"]
            }
            self._upload_document(doc, ARCEE_API_KEY, context_name)

    def _upload_document(self, doc, ARCEE_API_KEY, context_name):
        headers = {
            "Authorization": f"Bearer {ARCEE_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "context_name": context_name,
            "document": doc
        }

        response = requests.post("http://127.0.0.1:9001/v1/upload-context", headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise Exception(f"Failed to upload example. Response: {response.text}")