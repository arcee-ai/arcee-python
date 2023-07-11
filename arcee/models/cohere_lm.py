# cohere_lm.py
import cohere
import os
from cohere.custom_model_dataset import CsvDataset
from lm import LM

class CohereLM(LM):
    def __init__(self, model_name: str):
        super().__init__(model_name)
        
        self.cohere_api_key = os.getenv("COHERE_API_KEY")
        if self.cohere_api_key is None:
            raise ValueError("Missing environment variable to use the Cohere LM - retrieve yours at dashboard.cohere.ai: COHERE_API_KEY")
        self.client = cohere.Client(self.cohere_api_key)
        self.model = None

    def train(self, dataset_path: str):
        csv_data = CsvDataset(train_file=dataset_path, delimiter=",")
        self.model = self.client.create_custom_model(
            name=self.model_name,
            dataset=csv_data,
            model_type="GENERATIVE"
        )
        if self.model.status != "READY":
            raise Exception("Model is not ready")

    def predict(self, prompt: str):
        if self.model is None:
            raise Exception("You must train the model before making predictions")
        
        return self.client.generate(prompt=prompt, model=self.model.model_id)

