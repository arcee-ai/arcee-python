# DALM by Arcee

The Arcee client for executing domain-adpated language model routines

![Arcee DALMs](https://uploads-ssl.webflow.com/64c95915793c8a64a186e43e/64de2c7fb2c99494dec2e0a4_realistic-lifelike-dalmatian-dog-puppy-pastel-bright-vintage-outfits-commercial%201-min.jpg)


## Installation

```
pip install arcee-py
```

## Authenticating

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
arcee.upload_context("pubmed", doc_name="doc1", doc_text="whoa")
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
