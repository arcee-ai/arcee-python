import threading
import concurrent.futures
import os
from arcee.data import PromptSet
from random import randint
from tqdm import tqdm
from langchain.chat_models import ChatOpenAI
from arcee.data import InstructionSet
import csv

default_system_prompt = "You are the worlds best \
                        paraphraser for generating \
                        instruction examples to finetune \
                        an AI language model. I will be providing \
                        you with an instuction to generate \
                        variations of and you will be asked \
                        to generate variations of the instrcution."


def postprocess_variations(variation_response):
    # sometimes openai comes back with newlines even though it's been prompted not to
    variation_response = variation_response.replace("\n", "")

    return variation_response


lock = threading.Lock()


def generate_variations(
    chat,
    instruction_set,
    i,
    num_variations,
    num_context_examples,
    system_prompt,
    writer,
):
    example_samples_index = [
        randint(0, len(instruction_set) - 1) for p in range(0, num_context_examples)
    ]
    example_samples = [instruction_set[i] for i in example_samples_index]

    example = instruction_set.dataset[i]
    instruction = example["instruction"]
    response = example["response"]

    generation_prompt = (
        system_prompt
        + f"Given the following instuction: {instruction}"
        + "\n"
        + "and the following context of example intruction and responses. "
        + "\n".join(example_samples)
        + "\n"
        + f"generate {str(num_variations)}"
        + f" variations of the instruction {instruction} - separated by ###."
        + f"These variations should produce the same response: {response}"
        + "Please provide only one continguous string of the variations separated by ###."
        + "Your response will be split by ### afterwards and saved to a csv."
        + "Do not respond on new lines. And respond only with the the variation example without the "
        + f"instruction prefix {instruction_set.instruction_prefix}"
    )

    variation_response = postprocess_variations(chat.predict(generation_prompt))

    # write original examples
    with lock:
        writer.writerow([instruction, response])

        for variation in variation_response.split("###"):
            # strip space from openai response
            variation = variation.strip(" ")

            # avoid null variations
            if len(variation) > 0:
                writer.writerow([variation, response])


def generate_with_fixed_response(
    instruction_set: InstructionSet,
    num_variations,
    output_file,
    num_context_examples=5,
    system_prompt=default_system_prompt,
    parallelism=5,
    model_name="gpt-3.5-turbo",
):
    """
    Generate examples of input instructions with a fixed response and write them into a specified output file.

    Parameters:
    instruction_set (InstructionSet): An instance of InstructionSet that contains a set of instructions and responses.
    num_variations (int): The number of variations to generate.
    output_file (str): The name of the file to which the generated examples will be written.
    num_context_examples (int, optional): The number of context examples to provide during generation. Defaults to 5.
    system_prompt (str, optional): The system prompt to be used. Defaults to the value of default_system_prompt.
    parallism (int, optional): The number of parallel generations to use. Defaults to 5.
    model_name (str, optional): The name of the model to use. Defaults to "gpt-3.5-turbo".

    Returns:
    None
    """
    if model_name not in ["gpt-3.5-turbo", "gpt-4"]:
        raise ("Model name must be one of 'gpt-3.5-turbo' or 'gpt-4' for generation.")

    if not "OPENAI_API_KEY" in os.environ.keys():
        raise (
            Exception(
                "OPENAI_API_KEY environment variable must be set to use the generate method"
            )
        )

    chat = ChatOpenAI(temperature=0, model_name=model_name)

    if not isinstance(instruction_set, InstructionSet):
        raise TypeError(
            f"Expected `InstructionSet` but got `{type(instruction_set).__name__}` instead."
        )

    with open(output_file, "w", newline="") as file:
        writer = csv.writer(
            file, quoting=csv.QUOTE_ALL, quotechar='"', doublequote=True
        )
        writer.writerow(["instruction", "response"])

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
            with tqdm(total=len(instruction_set)) as pbar:
                for i in range(len(instruction_set)):
                    executor.submit(
                        generate_variations,
                        chat,
                        instruction_set,
                        i,
                        num_variations,
                        num_context_examples,
                        system_prompt,
                        writer,
                    )
                    pbar.update()
