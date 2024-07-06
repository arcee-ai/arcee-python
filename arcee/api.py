import csv
import os
from typing import Any, Dict, Generator, List, Literal, Optional, Union, cast

import yaml
from datasets import load_dataset
from requests import Response

from arcee import config
from arcee.api_handler import make_request, nonjson_request
from arcee.api_helpers import _chat_ml_messages_to_qa_pair
from arcee.dalm import check_model_status
from arcee.schemas.routes import Route


def upload_corpus_folder(corpus: str, s3_folder_url: str, tokenizer_name: str, block_size: int) -> Dict[str, str]:
    """
    Upload a corpus file to a context

    Args:
        corpus (str): The name of the corpus to upload to
        s3_folder_url (str): The S3 url of the file to upload
        tokenizer_name (str): The name of the tokenizer used for processing the corpus
        block_size (int): The block size used to pack the dataset in LLM continual pre-training,
        usually can be set to max_position_embeddings of the original model
    """

    if not s3_folder_url.startswith("s3://"):
        raise Exception("s3_folder_url must be an S3 url")

    data = {
        "corpus_name": corpus,
        "s3_folder_url": s3_folder_url,
        "tokenizer_name": tokenizer_name,
        "block_size": block_size,
    }

    return make_request("post", Route.pretraining + "/corpusUpload", data)


def upload_qa_pairs(
    qa_set: str, qa_pairs: List[Dict[str, str]], question_column: str = "question", answer_column: str = "answer"
) -> Dict[str, str]:
    """
    Upload a list of QA pairs to a specific QA set.

    Args:
        qa_set (str): The name of the QA set to upload to.
        qa_pairs (list): A list of dictionaries with keys "question" and "answer".

    Returns:
        Dict[str, str]: The response from the make_request call.
    """
    if len(qa_pairs) > 2000:
        raise Exception("You can only upload 2000 QA pairs at a time")

    qa_list = []
    for qa in qa_pairs:
        if question_column not in qa.keys() or answer_column not in qa.keys():
            raise Exception("Each QA pair must have a 'question' and an 'answer' key")

        qa_list.append({"question": qa[question_column], "answer": qa[answer_column]})

    data = {"qa_set_name": qa_set, "qa_pairs": qa_list}
    return make_request("post", Route.alignment + "/qaUpload", data)


def chunk_list(lst: List[Any], chunk_size: int) -> Generator[List[Any], Any, None]:
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


def upload_qa_pairs_from_csv(
    qa_set: str, csv_path: str, question_column: str = "question", answer_column: str = "answer", batch_size: int = 200
) -> None:
    """
    Upload QA pairs from a CSV file to a specific QA set.

    Args:
        qa_set (str): The name of the QA set to upload to.
        csv_path (str): The path to the CSV file containing QA pairs.
        question_column (str): The name of the column containing questions in the CSV file.
        answer_column (str): The name of the column containing answers in the CSV file.

    Returns:
        Dict[str, str]: The response from the make_request call.
    """
    qa_pairs = []

    print(f"Reading QA pairs from {csv_path}...")
    with open(csv_path, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if question_column not in row.keys() or answer_column not in row.keys():
                raise Exception(
                    f"Each row must have a '{question_column}' and an '{answer_column}' key."
                    + " You can override the column names using the question_column and answer_column arguments."
                )
            qa_pairs.append({f"{question_column}": row[question_column], f"{answer_column}": row[answer_column]})

    print(f"Total QA pairs read: {len(qa_pairs)}")
    # Split the QA pairs into chunks and upload each chunk separately
    for i, chunk in enumerate(chunk_list(qa_pairs, batch_size)):
        print(f"Uploading chunk {i + 1} of {len(qa_pairs) // batch_size + 1}...")

        upload_qa_pairs(qa_set=qa_set, qa_pairs=chunk, question_column=question_column, answer_column=answer_column)


def upload_hugging_face_dataset_qa_pairs(qa_set: str, hf_dataset_id: str, dataset_split: str, data_format: str) -> None:
    """
    Upload a list of QA pairs from a hugging face dataset to a specific QA set.

    NOTE: you will need to set HUGGINGFACE_TOKEN in your environment to use this function.

    Args:
        qa_set (str): The name of the QA set to upload to.
        hf_dataset_id (str): The HF dataset id (eg, org/dataset) that contains ChatML format in a 'messages' column.
        dataset_split (str): The name of the dataset split to use, eg, "train", "train_sft", etc..
        data_format (str): The format of the data in the dataset.
            Only "chatml" is currently supported, and it can only be single turn, not multi-turn.

    Returns:
        None
    """

    if data_format != "chatml":
        raise Exception(f"{data_format} not supported yet, only chatml is supported")

    qa_pairs = []

    # Load dataset from HF
    dataset = load_dataset(hf_dataset_id)

    # Convert the split to pandas
    df = dataset[dataset_split].to_pandas()

    # Loop over all the rows in df and convert the messages into QA pairs
    for i, row in df.iterrows():
        try:
            qa_pair_tuple = _chat_ml_messages_to_qa_pair(row["messages"])
            qa_pair = {"question": qa_pair_tuple[0], "answer": qa_pair_tuple[1]}
            qa_pairs.append(qa_pair)
        except Exception as e:
            print(f"Error on row {i}: {e}.  Skipping row")
            continue

    batch_size = 200

    print(f"Uploading {len(qa_pairs)} QA pairs in batches of {batch_size}")

    # Upload in chunks of batch_size
    for i in range(0, len(qa_pairs), batch_size):
        chunk = qa_pairs[i : i + batch_size]
        print(f"Uploading {batch_size} QA pairs..")
        upload_qa_pairs(qa_set, chunk)

    print("Finished uploading QA pairs")


def upload_docs(context: str, docs: List[Dict[str, str]]) -> Dict[str, str]:
    """
    Upload a list of documents to a context

    Args:
        context (str): The name of the context to upload to
        docs (list): A list of dictionaries with keys "doc_name" and "doc_text"

        Any other keys in the `docs` will be assumed as metadata, and will be uploaded as such. This metadata can
            be filtered on during retrieval and generation.
    """
    doc_list = []
    for doc in docs:
        if "doc_name" not in doc.keys() or "doc_text" not in doc.keys():
            raise Exception("Each document must have a doc_name and doc_text key")

        new_doc: Dict[str, Union[str, Dict]] = {"name": doc.pop("doc_name"), "document": doc.pop("doc_text")}
        # Any other keys are metadata
        if doc:
            new_doc["meta"] = doc
        doc_list.append(new_doc)

    data = {"context_name": context, "documents": doc_list}
    return make_request("post", Route.contexts, data)


def start_pretraining(
    pretraining_name: str,
    corpus: str,
    base_model: str,
    target_compute: Optional[str] = None,
    capacity_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Start pretraining a model

    Args:
        pretraining_name (str): The name of the pretraining job.
        corpus (str): The name of the corpus to use.
        base_model (str): The name of the base model to use.
        target_compute (Optional[str]): The name of the compute to use,
            e.g., "g5.2xlarge" or "capacity". If omitted, the default
            compute will be used.
        capacity_id (Optional[str]): The name of the capacity block ID
            to use. If omitted, an instance will be launched to perform
            training.
    """

    data = {"pretraining_name": pretraining_name, "corpus_name": corpus, "base_model": base_model}

    if target_compute:
        data["target_compute"] = target_compute

    if capacity_id:
        data["capacity_id"] = capacity_id

    return make_request("post", Route.pretraining + "/startTraining", data)


def mergekit_yaml(
    merging_name: str, merging_yaml_path: str, target_compute: Optional[str] = None, capacity_id: Optional[str] = None
) -> Dict[str, str]:
    """
    Start merging models

    Args:
        merging_name (str): The name of the merging job.
        merging_yaml (str): The yaml file containing the merging
            instructions - https://github.com/arcee-ai/mergekit/tree/main/examples.
        target_compute (Optional[str]): The name of the compute to use,
            e.g., "g5.2xlarge" or "capacity". If omitted, the default
            compute will be used.
        capacity_id (Optional[str]): The name of the capacity block ID
            to use. If omitted, an instance will be launched to perform
            training.
    """

    if not merging_yaml_path.endswith(".yaml"):
        raise Exception("The merging yaml file must be a .yaml file")

    if not os.path.exists(merging_yaml_path):
        raise Exception(f"The merging yaml file {merging_yaml_path} does not exist")

    with open(merging_yaml_path, "r") as file:
        merging_yaml = yaml.safe_load(file)

        data = {"merging_name": merging_name, "best_merge_yaml": str(merging_yaml)}

        if target_compute:
            data["target_compute"] = target_compute

        if capacity_id:
            data["capacity_id"] = capacity_id

        return make_request("post", Route.merging + "/start", data)


def mergekit_evolve(
    merging_name: str,
    arcee_aligned_models: Optional[List[str]] = None,
    arcee_merged_models: Optional[List[str]] = None,
    arcee_pretrained_models: Optional[List[str]] = None,
    hf_models: Optional[List[str]] = None,
    arcee_eval_qa_set_names_and_weights: Optional[List[dict]] = None,
    general_evals_and_weights: Optional[List[dict]] = None,
    base_model: Optional[str] = None,
    merge_method: Optional[str] = "ties",
    target_compute: Optional[str] = None,
    capacity_id: Optional[str] = None,
    time_budget_secs: int = 3600,
) -> Dict[str, str]:
    """
    Start merging models

    Args:
        merging_name (str): The name of the merging job
        arcee_aligned_models (list): A list of Arcee models to merge
        arcee_merged_models (list): A list of Arcee models already merged
        arcee_pretrained_models (list): A list of pretrained Arcee models
        hf_models (list): A list of Hugging Face models to merge
        eval_qa_set_names_and_weights (list): A list of QA set names to merge
        general_evals_and_weights (list): A list of general evaluations to merge
        base_model (str): The name of the base model to use
        merge_method (str): The merging method to use - https://github.com/arcee-ai/mergekit/blob/main/mergekit/merge_methods/__init__.py
        time_budget_secs (int): The time budget for the merging job (seconds) - the evolution will stop after this time
    """

    if general_evals_and_weights is None:
        general_evals_and_weights = [
            {"agieval_gaokao_physics": 1, "agieval_gaokao_english": 1, "agieval_logiqa_en": 1, "truthfulqa_gen": 1}
        ]

    data = {
        "merging_name": merging_name,
        "arcee_aligned_models": arcee_aligned_models,
        "arcee_merged_models": arcee_merged_models,
        "arcee_pretrained_models": arcee_pretrained_models,
        "hf_models": hf_models,
        "arcee_eval_qa_set_names_and_weights": arcee_eval_qa_set_names_and_weights,
        "general_evals_and_weights": general_evals_and_weights,
        "base_model": base_model,
        "merge_method": merge_method,
        "target_compute": target_compute,
        "capacity_id": capacity_id,
        "time_budget_secs": time_budget_secs,
    }

    return make_request("post", Route.merging + "/start", data)


def delete_corpus(corpus: str) -> Dict[str, str]:
    """
    Delete a corpus

    Args:
        corpus (str): The name of the corpus to delete
    """

    data = {"corpus_name": corpus}

    return make_request("post", Route.pretraining + "/deleteCorpus", data)


def corpus_status(corpus: str) -> Dict[str, str]:
    """
    Check the status of a corpus

    Args:
        corpus (str): The name of the corpus to check the status
    """

    data = {"corpus_name": corpus}

    return make_request("post", Route.pretraining + "/corpus/status", data)


def start_alignment(
    alignment_name: str,
    qa_set: str,
    pretrained_model: Optional[str] = None,
    merging_model: Optional[str] = None,
    alignment_model: Optional[str] = None,
    target_compute: Optional[str] = None,
    capacity_id: Optional[str] = None,
) -> Dict[str, str]:
    """
    Start the alignment of a model.

    Args:
        alignment_name (str): The name of the alignment job.
        qa_set (str): The name of the QA set to use.
        pretrained_model (Optional[str]): The name of the pretrained model to use, if any.
        merging_model (Optional[str]): The name of the merging model to use, if any.
        alignment_model (Optional[str]): The name of the final alignment model to use, if any.
        target_compute (Optional[str]): The name of the compute to use, e.g., "g5.2xlarge" or
            "capacity". If omitted, the default compute will be used.
        capacity_id (Optional[str]): The name of the capacity block ID to use. If omitted, an
            instance will be launched to perform training.
    """

    data = {
        "alignment_name": alignment_name,
        "qa_set_name": qa_set,
        "pretrained_model": pretrained_model,
        "merging_model": merging_model,
        "alignment_model": alignment_model,
    }

    if target_compute:
        data["target_compute"] = target_compute

    if capacity_id:
        data["capacity_id"] = capacity_id

    # Assuming make_request is a function that handles the request, it's called here
    return make_request("post", Route.alignment + "/startAlignment", data)


def upload_alignment(alignment_name: str, alignment_id: str, qa_set_id: str, pretraining_id: str) -> Dict[str, str]:
    data = {
        "alignment_name": alignment_name,
        "alignment_id": alignment_id,
        "qa_set_id": qa_set_id,
        "pretraining_id": pretraining_id,
    }
    return make_request("post", Route.alignment + "/uploadAlignment", data)


def start_retriever_training(name: str, context: str) -> None:
    data = {"name": name, "context": context}
    make_request("post", Route.train_model, data)
    org = get_current_org()
    status_url = f"{config.ARCEE_APP_URL}/{org}/models/{name}/training"
    print(
        f'Retriever model training started - view model status at {status_url} \
          or with arcee.get_retriever_status("{name}")'
    )


def get_retriever_status(id_or_name: str) -> Dict[str, str]:
    """Gets the status of a retriever training job"""
    return check_model_status(id_or_name)


def start_deployment(
    deployment_name: str,
    alignment: Optional[str] = None,
    merging: Optional[str] = None,
    pretraining: Optional[str] = None,
    retriever: Optional[str] = None,
    target_instance: Optional[str] = None,
    openai_compatability: Optional[bool] = False,
) -> Dict[str, str]:
    data = {
        "deployment_name": deployment_name,
        "alignment_name": alignment,
        "merging_name": merging,
        "pretraining_name": pretraining,
        "retriever_name": retriever,
        "target_instance": target_instance,
        "openai_compatability": openai_compatability,
    }
    return make_request("post", Route.deployment + "/startDeployment", data)


def stop_deployment(deployment_name: str) -> Dict[str, str]:
    data = {"deployment_name": deployment_name}
    return make_request("post", Route.deployment + "/stopDeployment", data)


def generate(deployment_name: str, query: str) -> Dict[str, str]:
    data = {"deployment_name": deployment_name, "query": query}
    return make_request("post", Route.deployment + "/generate", data)


def retrieve(deployment_name: str, query: str, size: Optional[int] = 5) -> Dict[str, str]:
    data = {"deployment_name": deployment_name, "query": query, "size": size}
    return make_request("post", Route.deployment + "/retrieve", data)


def embed(deployment_name: str, query: str) -> Dict[str, str]:
    data = {"deployment_name": deployment_name, "query": query}
    return make_request("post", Route.deployment + "/embed", data)


def get_current_org() -> str:
    return make_request("get", Route.identity)["org"]


def list_pretrainings() -> List[Dict[str, str]]:
    return cast(List[Dict[str, str]], make_request("get", Route.pretraining + "/"))


model_weight_types = Literal["pretraining", "alignment", "retriever", "merging"]

type_to_weights_route = {
    "pretraining": Route.pretraining + "/{id_or_name}/weights",
    "alignment": Route.alignment + "/{id_or_name}/weights",
    "retriever": Route.retriever + "/{id_or_name}/weights",
    "merging": Route.merging + "/{id_or_name}/weights",
}


def download_weights(type: model_weight_types, id_or_name: str) -> Response:
    """
    Download the weights of a trained model on the Arcee platform.

    type: The type of model to download weights for.
        Can be one of "pretraining", "alignment", "retriever", or "merging".
    id_or_name: The ID or name of the model to download weights for.
    """
    route = type_to_weights_route[type].format(id_or_name=id_or_name)
    return nonjson_request("get", route, stream=True)
