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
```

## Train DALM

Train a DALM with the context you have uploaded.

```
import arcee
dalm = arcee.train_dalm("medical_dalm", context="pubmed")
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
