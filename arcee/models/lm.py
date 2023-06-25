from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from arcee.data import PromptSet
from arcee.data import InstructionSet

question_prompt = "### Question: "
answer_prompt = "### Answer: "

def formatting_prompts_func(example):
    text = f"{question_prompt} {example['instruction']} {answer_prompt} {example['response']}"
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
  
    def train(self, dataset_path, epochs=3, peft=False, output_dir=None):
        
        instruction_set = InstructionSet(dataset_path)
        print("Number of training examples: " + str(len(instruction_set.dataset)))
        
        #prompt_set = PromptSet(dataset)
        
        #hf_dataset = load_dataset("csv", data_files=dataset, split="train")
        
        if output_dir is None:
            output_dir = self.name + "-" + dataset_path.split("/")[-1]
        
        training_args = TrainingArguments(
            num_train_epochs=epochs,
            output_dir=output_dir,
        )
        
        if peft:
            print("Training with PEFT...")
            self.trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=instruction_set.dataset,
                dataset_text_field="text",
                max_seq_length=32,
                peft_config=self.peft_config,
                packing=False,
            )
        else:
            self.trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=instruction_set.dataset,
                dataset_text_field="text",
                max_seq_length=32,
                packing=False
                #formatting_func=formatting_prompts_func,
                # peft_config=self.peft_config,
                #packing=True,
            )
            
        self.trainer.train()

    def predict(self, prompt,  max_length=100, min_length=50, temperature=0.1):
        prompt = f"{question_prompt} {prompt}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, max_length=max_length, min_length=min_length, temperature=temperature)
        prediction = self.tokenizer.decode(outputs[0])
        #find text between answer prompt and <|endoftext|>
        #answer = generation.split(answer_prompt)[1].replace("<|endoftext|>", "").strip()
        return prediction
        