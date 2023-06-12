from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig


def formatting_prompts_func(example):
    text = f"### Question: {example['question']}\n ### Answer: {example['answer']}"
    return text

class LM():
    """General language model class"""
    
    def __init__(self, name):
        self.name = name
        
        self.peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(name)
        
        self.tokenizer = AutoTokenizer.from_pretrained(name)
  
    def train(self, dataset):
        dataset = load_dataset("csv", data_files=dataset, split="train")
        print(dataset)
        # dataset = load_dataset("imdb", split="train")
        
        self.trainer = SFTTrainer(
            self.model,
            train_dataset=dataset,
            dataset_text_field="text",
            max_seq_length=512,
            peft_config=self.peft_config,
            packing=True
        )
        
        self.trainer.train()


    def predict(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids)
        return self.tokenizer.decode(outputs[0])
        
        
        
    