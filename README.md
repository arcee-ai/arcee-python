# DALM by Arcee

The Arcee client for executing domain-adpated language model routines

![Arcee DALMs](https://uploads-ssl.webflow.com/64c95915793c8a64a186e43e/64de2c7fb2c99494dec2e0a4_realistic-lifelike-dalmatian-dog-puppy-pastel-bright-vintage-outfits-commercial%201-min.jpg)


## Installation

```
pip install arcee-py
```

## Authenticating

Your Arcee API key is obtained at app.arcee.ai

In bash:

```
export ARCEE_API_KEY=********
```

In notebook:

```
import os
os.environ["ARCEE_API_KEY"] = "********"
```

## Upload Context

Upload context for your domain adapted langauge model to draw from.

```
import arcee
arcee.upload_doc("pubmed", doc_name="doc1", doc_text="whoa")
# or
# arcee.upload_docs("pubmed", docs=[{"doc_name": "doc1", "doc_text": "foo"}, {"doc_name": "doc2", "doc_text": "bar"}])
```

## Train DALM

Train a DALM with the context you have uploaded.

```
import arcee
dalm = arcee.train_dalm("medical_dalm", context="pubmed")
# Wait for training to complete
arcee.get_dalm_status("medical_dalm")
```

The DALM training procedure trains your model in context and stands up an index for your model to draw from.

## DALM Generation

```
import arcee
med_dalm = arcee.get_dalm("medical_dalm")
med_dalm.generate("What are the components of Scoplamine?")
```

## DALM Retrieval

Retrieve documents for a given query and to view them or plug into a different LLM.

```
import arcee
med_dalm = arcee.get_dalm("medical_dalm")
med_dalm.retrieve("my query")
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
1. Open a PR increasing the `__version__` of arcee-py. You can manually edit it or run `inv uv`
2. Create a new release, with the name being the `__version__` of arcee-py

### Manual release [not recommended]

We do not recommend this. If you need to, please make the version number an alpha or beta release.<br>
If you need to create a manual release, you can run `inv build && inv publish`
