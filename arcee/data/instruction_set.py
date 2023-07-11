from datasets import load_dataset
import os
import requests
import json

# reproducable dataset splits
import random

random.seed(777)


class InstructionSet:
    """
    An instruction set is a collection of instructions and responses that is used to instruction finetune an LM.
    """

    def __init__(
        self,
        data_file,
        instruction_prefix="### Instruction:",
        response_prefix="### Response:",
    ):
        if not data_file.endswith(".csv"):
            raise (Exception("Datafile must be a .csv file"))

        self.dataset = load_dataset("csv", data_files=data_file, split="train")
        dataset_fields = list(self.dataset.features.keys())
        self.dataset_fields = dataset_fields

        if "prompt" not in dataset_fields:
            raise (Exception("Datafile must contain a column named 'prompt'"))

        if "completion" not in dataset_fields:
            raise (Exception("Datafile must contain a column named 'completion'"))

        self.instruction_prefix = instruction_prefix
        self.response_prefix = response_prefix

        # TODO refactor for larger datasets when TRL formatting_func is fixed

        text_column = [
            self.instruction_prefix + str(a) + " " + self.response_prefix + str(b)
            for a, b in zip(self.dataset["prompt"], self.dataset["completion"])
        ]
        self.dataset = self.dataset.add_column("text", text_column)

        train_test_split_ratio = 0.2
        dataset_train_val = self.dataset.train_test_split(
            test_size=train_test_split_ratio
        )

        self.train_dataset = dataset_train_val["train"]
        self.val_dataset = dataset_train_val["test"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """
        This function formats an entry as it will appear during finetuning
        """

        return f"{self.instruction_prefix} {self.dataset[idx]['prompt']} {self.response_prefix} {self.dataset[idx]['completion']}"

    def formatting_prompts_func(self, example):
        """
        This function takes a dataset example and formats it into a prompt for finetuning.
        """

        return f"{self.instruction_prefix} {example['prompt']} {self.response_prefix} {example['completion']}"

    def upload(self, instruction_set_name):
        ARCEE_API_KEY = os.getenv("ARCEE_API_KEY")
        if ARCEE_API_KEY is None:
            raise Exception("ARCEE_API_KEY must be in the environment")

        for i in range(len(self)):
            example = {
                "prompt": self.dataset[i]["prompt"],
                "completion": self.dataset[i]["completion"]
            }
            self._upload_single_example(example, ARCEE_API_KEY, instruction_set_name)

    def _upload_single_example(self, example, ARCEE_API_KEY, instruction_set_name):
        headers = {
            "Authorization": f"Bearer {ARCEE_API_KEY}",
            "Content-Type": "application/json"
        }
        data = {
            "instruction_set_name": instruction_set_name,
            "example": example
        }
        response = requests.post("http://127.0.0.1:9001/v1/upload-example", headers=headers, data=json.dumps(data))

        if response.status_code != 200:
            raise Exception(f"Failed to upload example. Response: {response.text}")