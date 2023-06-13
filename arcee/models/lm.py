from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from arcee.data import PromptSet


question_prompt = "### Question: "
answer_prompt = "### Answer: "

def formatting_prompts_func(example):
    text = f"{question_prompt} {example['question']} {answer_prompt} {example['answer']}"
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
  
    def train(self, dataset, epochs=3, peft=False, output_dir=None):
        
        prompt_set = PromptSet(dataset)
        
        if output_dir is None:
            output_dir = self.name + "-" + dataset.split("/")[-1]
        # dataset = load_dataset("imdb", split="train")
        
        # training_args = TrainingArguments(
        #     output_dir=args.output_dir,
        #     dataloader_drop_last=True,
        #     evaluation_strategy="steps",
        #     max_steps=args.max_steps,
        #     eval_steps=args.eval_freq,
        #     save_steps=args.save_freq,
        #     logging_steps=args.log_freq,
        #     per_device_train_batch_size=args.batch_size,
        #     per_device_eval_batch_size=args.batch_size,
        #     learning_rate=args.learning_rate,
        #     lr_scheduler_type=args.lr_scheduler_type,
        #     warmup_steps=args.num_warmup_steps,
        #     gradient_accumulation_steps=args.gradient_accumulation_steps,
        #     gradient_checkpointing=not args.no_gradient_checkpointing,
        #     fp16=not args.no_fp16,
        #     bf16=args.bf16,
        #     weight_decay=args.weight_decay,
        #     run_name="llama-7b-finetuned",
        #     report_to="wandb",
        #     ddp_find_unused_parameters=False,
        # )
        
        training_args = TrainingArguments(
            num_train_epochs=epochs,
            output_dir=output_dir,
        )
        
        if peft:
            print("Training with PEFT...")
            self.trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=prompt_set.dataset,
                dataset_text_field="text",
                max_seq_length=512,
                formatting_func=prompt_set.formatting_prompts_func,
                peft_config=self.peft_config,
                packing=True,
            )
        else:
            self.trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=prompt_set.dataset,
                dataset_text_field="text",
                max_seq_length=512,
                formatting_func=prompt_set.formatting_prompts_func,
                # peft_config=self.peft_config,
                packing=True,
            )
            
        self.trainer.train()


    def generate(self, prompt,  max_length=100, min_length=50, temperature=0.1):
        prompt = f"{question_prompt} {prompt}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(input_ids, max_length=max_length, min_length=min_length, temperature=temperature)
        generation = self.tokenizer.decode(outputs[0])
        #find text between answer prompt and <|endoftext|>
        #answer = generation.split(answer_prompt)[1].replace("<|endoftext|>", "").strip()
        return generation
        

        
    