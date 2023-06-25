# Arcee

The open source toolkit for finetuning and deploying LLMs

### Finetune LLMS

```
from arcee.models import LM
from arcee.data import Instuctions

lm = LM("falcon30b")
instructions = Instructions("./datasets/stripe-api.json")
lm.train(instructions)

lm.predict("Place an order for the LLM-9000 product for 100 USD to the card 3007200039992000")
```

### Deploy LLMS

Authenticate

```
import arcee
arcee.login()
```

Deploy to the Arcee cloud

```
project = arcee.create_project("stripe-api-operator")
#project = arcee.load_project(...)

#regulated under 7b params for free
#only PEFT uploadable for free

hosted_lm = project.deploy(llm)
hosted_lm.url
hosted_lm.predict("Place an order for the LLM-9000 product for 100 USD to the card 3007200039992000")

#view a streaming web app of the llm
project.demo()
```

### LangChain Integration

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


