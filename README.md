# Arcee Python Client

> The Arcee Python client allows you to manage CPT, SFT, DPO, and Merge models on the Arcee Platform.

This client may be used as a CLI by invoking `arcee` from the terminal, or as an SDK for programmatic use by `import arcee` in Python.

Learn more at https://docs.arcee.ai

## Installation

```
pip install --upgrade arcee-py
```

## Authenticating

Your Arcee API key is obtained at https://app.arcee.ai

In bash:

```
export ARCEE_API_KEY=********
```

In notebook:

```
import os
os.environ["ARCEE_API_KEY"] = "********"
```

(Optional) To customize the URL of the Arcee platform:

```
export ARCEE_API_URL="https://your-url.arcee.ai"
```

(Optional) To specify an organization to issue requests for:

```
export ARCEE_ORG="my-organization"
```

If you do not specify an organization, your default organization will be used. You can change the default in your Arcee account settings.

## Upload Context

Upload context for retriever training:

```
import arcee
arcee.upload_docs("pubmed", docs=[{"doc_name": "doc1", "doc_text": "foo"}, {"doc_name": "doc2", "doc_text": "bar"}])
```

## Upload Finetuning Dataset

### Method 1: Via CSV

```
arcee.upload_instructions_from_csv(
  "finetuning-dataset-name",
  csv_path="./your_data.csv",
  prompt_column="prompt",
  completion_column="completion"
)
```

### Method 2: Via HF Dataset

NOTE: you will need to set `HUGGINGFACE_TOKEN` in your environment to use this function.


```
arcee.api.upload_hugging_face_dataset_qa_pairs(
    "my_qa_pairs",
    hf_dataset_id="org/dataset",
    dataset_split="train",
    data_format="chatml"
)
```

## Using the Arcee CLI

You can easily train and use your Domain-Adapted Language Model (DALM) with Arcee using the CLI. Follow these steps post installation to train and utilize your DALM:

### Upload Context

Upload a context file for your DALM like,
```shell
arcee upload context pubmed --file doc1
```
Upload all files in a directory like,
```shell
arcee upload context pubmed --directory docs
```
Upload any combination of files and directories with,
```shell
arcee upload context pubmed --directory some_docs --file doc1 --directory more_docs --file doc2
```
*Note: The upload command ensures only valid and unique files are uploaded.*

### Train your DALM:
Train your DALM with any uploaded context like,
```shell
arcee train medical_dalm --context pubmed
# wait for training to complete...
```
### DALM Generation:
Generate text completions from a model like,
 ```shell
arcee generate medical_dalm --query "Can AI-driven music therapy contribute to the rehabilitation of patients with disorders of consciousness?"
```

### DALM Retrieval:
Retrieve documents for a given query and to view them or plug into a different LLM like,
```shell
arcee retrieve medical_dalm --query "Can AI-driven music therapy contribute to the rehabilitation of patients with disorders of consciousness?"
```

# Contributing

We use `invoke` to manage this repo. You don't need to use it, but it simplifies the workflow.
## Set up the repo
```shell
git clone https://github.com/arcee-ai/arcee-python && cd arcee-python
# optionally setup your virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate
# install repo
pip install invoke
inv install
```

## Format, lint, test
```shell
inv format  # run black and ruff
inv lint    # black check, ruff check, mypy
inv test    # pytest
```

## Publishing
We publish in this repo by creating a new release/tag in github. On release, a github action will
publish the `__version__` of arcee-py that is in `arcee/__init__.py`

**So you need to increase that version before releasing, otherwise it will fail**

### To create a new release
1. Open a PR increasing the `__version__` of arcee-py. You can manually edit it or run `inv version`
2. Create a new release, with the name being the `__version__` of arcee-py

### Manual release [not recommended]

We do not recommend this. If you need to, please make the version number an alpha or beta release.<br>
If you need to create a manual release, you can run `inv build && inv publish`
