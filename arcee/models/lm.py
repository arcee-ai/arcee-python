from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from arcee.data import InstructionSet

question_prompt = "### Question: "
answer_prompt = "### Answer: "


def formatting_prompts_func(example):
    text = (
        f"{question_prompt} {example['prompt']} {answer_prompt} {example['completion']}"
    )
    return text


class LM:
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
        eos_token_id = self.tokenizer.eos_token_id
        self.eos_token = self.tokenizer.decode([eos_token_id])

    def train(self, dataset_path, epochs=10, peft=False, output_dir=None, batch_size=8):
        instruction_set = InstructionSet(dataset_path)
        # set instruction set for decoding

        # TODO: refactor for reloading
        self.instruction_set = instruction_set

        print("Number of training examples: " + str(len(instruction_set.dataset)))

        # prompt_set = PromptSet(dataset)

        # hf_dataset = load_dataset("csv", data_files=dataset, split="train")

        if output_dir is None:
            output_dir = self.name + "-" + dataset_path.split("/")[-1]

        training_args = TrainingArguments(
            num_train_epochs=epochs,
            # max_steps=max_steps,
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch",
        )

        if peft:
            print("Training with PEFT...")
            self.trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=instruction_set.train_dataset,
                eval_dataset=instruction_set.val_dataset,
                dataset_text_field="text",
                peft_config=self.peft_config,
                packing=True,
            )
        else:
            self.trainer = SFTTrainer(
                self.model,
                args=training_args,
                train_dataset=instruction_set.dataset,
                eval_dataset=instruction_set.val_dataset,
                dataset_text_field="text",
                packing=True,
            )

        self.trainer.train()

    def postprocess_prediction(self, prediction):
        return (
            prediction.split(self.instruction_set.response_prefix)[1]
            .replace(self.eos_token, "")
            .strip()
        )

    def predict(self, prompt, max_length=100, min_length=50, temperature=0.1):
        prompt = f"{question_prompt} {prompt}"
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
        )
        prediction = self.tokenizer.decode(outputs[0])
        # find text between answer prompt and <|endoftext|>
        # answer = generation.split(answer_prompt)[1].replace("<|endoftext|>", "").strip()
        try:
            return self.postprocess_prediction(prediction)
        except:
            return prediction
