# Arcee

:tulip:	The open source alignment toolkit for finetuning and deploying LLMs :tulip:

## Finetuning Datasets

The Arcee toolkit contains routines for you to manage and generate finetuning datasets. The arcee toolkit supports supervised finetuning (SFT) aka intruction tuning. 

### ✍️ Instruction Set ✍️

Instruction tuning datasets contain a series of prompt and completion examples.

Instruction sets can be loaded from a csv.

```
from arcee.data import InstructionSet
instruction_set = InstructionSet("./datasets/strip_api.csv")
```

Or downloaded from the Arcee platform

```
from arcee.data import InstructionSet
instruction_set = InstructionSet("https://app.arcee.ai/arcee/alpaca")
```

### Self-Instruct Generation

### Evolv-Instruct Genertation

### Explain-Instruct Generation


## Finetuning Models

### HuggingFace

```
from arcee.models import LM
from arcee.data import Instuctions

lm = LM("falcon30b")
instructions = Instructions("./datasets/stripe-api.json")
lm.train(instructions)

lm.predict("Place an order for the LLM-9000 product for 100 USD to the card 3007200039992000")
```

### OpenAI

### Cohere

### Together

### Mosaic ML

## LangChain Integration

```
from langchain import Arcee
#goes in llms/arcee.py

prompt_template = "Write a stripe API request for the following: {order}."

llm = Arcee(temperature=0)
llm_chain = LLMChain(
    llm=llm,
    prompt=PromptTemplate.from_template(prompt_template)
)
llm_chain("Place an order for the LLM-9000 product for 100 USD to the card 3007200039992000")
```

## Domain Pretraining

Coming soon!

## 🗻 Arcee Platform 🗻

Arcee offers a platform for managing your proprietary language models in production. We offer a version of our platform hosted in our cloud as well as a deployable version that you can take on prem.

### Authenticaiton

To use the arcee platform, visit https://app.arcee.com/api-settings and `export ARCEE_API_KEY=*******` in your environment.

### Dataset Managemnt

View, search, and edit datasets. 

```
instruction_set.upload("project-name")
```

### Hosted Training

### Hosted Inference

### Lifecycle Management




