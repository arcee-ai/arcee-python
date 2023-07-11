# huggingface.py
from datasets import load_dataset
from trl import SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from arcee.data import InstructionSet
from arcee.models.lm import LM

class HFLM(LM):
    """HuggingFace-based language model class"""

    def __init__(self, model_name: str, peft_config: LoraConfig = None):
        self.model_name = model_name
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.peft_config = peft_config if peft_config is not None else LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self.trainer = None
        self.eos_token = self.tokenizer.decode([self.tokenizer.eos_token_id])

    def train(
        self,
        dataset_path: str,
        output_dir: str = None,
        epochs: int = 10,
        use_peft: bool = False,
        batch_size: int = 8,
    ):
        instruction_set = InstructionSet(dataset_path)
        self.instruction_set = instruction_set

        print(f"Number of training examples: {len(instruction_set.dataset)}")

        if output_dir is None:
            output_dir = f"{self.model_name}-{dataset_path.split('/')[-1]}"

        training_args = TrainingArguments(
            num_train_epochs=epochs,
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            evaluation_strategy="epoch",
        )

        self.trainer = SFTTrainer(
            self.model,
            args=training_args,
            train_dataset=instruction_set.train_dataset,
            eval_dataset=instruction_set.val_dataset,
            dataset_text_field="text",
            peft_config=self.peft_config if use_peft else None,
            packing=True,
        )

        self.trainer.train()

    def postprocess_prediction(self, prediction):
        return (
            prediction.split(self.instruction_set.response_prefix)[1]
            .replace(self.eos_token, "")
            .strip()
        )

    def predict(
        self,
        prompt: str,
        max_length: int = 100,
        min_length: int = 50,
        temperature: float = 0.1,
    ):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = self.model.generate(
            input_ids,
            max_length=max_length,
            min_length=min_length,
            temperature=temperature,
        )
        prediction = self.tokenizer.decode(outputs[0])
        try:
            return self.postprocess_prediction(prediction)
        except Exception:
            return prediction