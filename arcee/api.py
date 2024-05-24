from typing import Dict, List, Optional, Union

from arcee import config
from arcee.api_handler import make_request
from arcee.dalm import DALM, check_model_status
from arcee.schemas.routes import Route
import csv
import yaml
import os


def upload_corpus_folder(corpus: str, s3_folder_url: str) -> Dict[str, str]:
    """
    Upload a corpus file to a context

    Args:
        corpus (str): The name of the corpus to upload to
        file_s3_url (str): The S3 url of the file to upload
    """

    if not s3_folder_url.startswith("s3://"):
        raise Exception("folder_s3_url must be an S3 url")

    data = {"corpus_name": corpus, "s3_folder_url": s3_folder_url}

    return make_request("post", Route.pretraining + "/corpusUpload", data)

def upload_qa_pairs(qa_set: str, qa_pairs: List[Dict[str, str]], prompt_column: str = "prompt", completion_column: str = "completion") -> Dict[str, str]:
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
        if prompt_column not in qa.keys() or completion_column not in qa.keys():
            raise Exception("Each QA pair must have a 'question' and an 'answer' key")

        qa_list.append({"question": qa[prompt_column], "answer": qa[completion_column]})

    data = {"qa_set_name": qa_set, "qa_pairs": qa_list}
    return make_request("post", Route.alignment + "/qaUpload", data)

def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def upload_instructions_from_csv(qa_set: str, csv_path: str, prompt_column: str = "prompt", completion_column: str = "completion", batch_size: int = 200) -> None:
    """
    Upload QA pairs from a CSV file to a specific QA set.

    Args:
        qa_set (str): The name of the QA set to upload to.
        csv_path (str): The path to the CSV file containing QA pairs.

    Returns:
        Dict[str, str]: The response from the make_request call.
    """
    qa_pairs = []

    print(f"Reading QA pairs from {csv_path}...")
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if prompt_column not in row.keys() or completion_column not in row.keys():
                #raise Exception(f"Each row must have a '{question_column}' and an '{answer_column}' key. You can override the column names using the question_column and answer_column arguments.")
                raise Exception(f"Each row must have a '{prompt_column}' and an '{completion_column}' key. You can override the column names using the prompt_column and completion_column arguments.")
            qa_pairs.append({
                f"{prompt_column}": row[prompt_column],
                f"{completion_column}": row[completion_column]
            })

    print(f"Total QA pairs read: {len(qa_pairs)}")
    # Split the QA pairs into chunks and upload each chunk separately
    for i, chunk in enumerate(chunk_list(qa_pairs, batch_size)):
        print(f"Uploading chunk {i + 1} of {len(qa_pairs) // batch_size + 1}...")
        
        upload_qa_pairs(qa_set=qa_set, qa_pairs=chunk, prompt_column=prompt_column, completion_column=completion_column)


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


def start_pretraining(pretraining_name: str, corpus: str, base_model: str) -> None:
    """
    Start pretraining a model

    Args:
        pretraining_name (str): The name of the pretraining job
        corpus (str): The name of the corpus to use
        base_model (str): The name of the base model to use
    """

    data = {"pretraining_name": pretraining_name, "corpus_name": corpus, "base_model": base_model}

    return make_request("post", Route.pretraining + "/startTraining", data)

def mergekit_yaml(
    merging_name: str,
    merging_yaml_path: str
) -> None:
    """
    Start merging models

    Args:
        merging_name (str): The name of the merging job
        merging_yaml (str): The yaml file containing the merging instructions
    """

    if not merging_yaml_path.endswith(".yaml"):
        raise Exception("The merging yaml file must be a .yaml file")

    if not os.path.exists(merging_yaml_path):
        raise Exception(f"The merging yaml file {merging_yaml_path} does not exist")

    with open(merging_yaml_path, 'r') as file:
        merging_yaml = yaml.safe_load(file)
    
        data = {
            "merging_name": merging_name,
            "best_merge_yaml": str(merging_yaml)
        }

        return make_request("post", Route.merging + "/startMerging", data)

def mergekit_evolve(
    merging_name: str,
    wandb_key: str = None,
    arcee_aligned_models: Optional[List[str]] = None,
    arcee_merged_models: Optional[List[str]] = None,
    arcee_pretrained_models: Optional[List[str]] = None,
    hf_models: Optional[List[str]] = None,
    arcee_eval_qa_set_names_and_weights: Optional[List[dict]] = None,
    general_evals_and_weights: Optional[List[dict]] = None,
    base_model: Optional[str] = None,
    merge_method: Optional[str] = None,
    target_compute: str = None,
    capacity_id: str = None,
    time_budget_secs: int = 1,
) -> None:
    """
    Start merging models

    Args:
        merging_name (str): The name of the merging job
        arcee_aligned_models (list): A list of ARCEE models to merge
        arcee_merged_models (list): A list of ARCEE models already merged
        arcee_pretrained_models (list): A list of pretrained ARCEE models
        hf_models (list): A list of Hugging Face models to merge
        eval_qa_set_names_and_weights (list): A list of QA set names to merge
        general_evals_and_weights (list): A list of general evaluations to merge
        base_model (str): The name of the base model to use
        merge_method (str): The merging method to use
        time_budget_secs (int): The time budget for the merging job (seconds)
    """
    
    data = {
        "merging_name": merging_name,
        "wandb_key": wandb_key,
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

    return make_request("post", Route.merging + "/startMerging", data)


def delete_corpus(corpus: str) -> None:
    """
    Delete a corpus

    Args:
        corpus (str): The name of the corpus to delete
    """

    data = {"corpus_name": corpus}

    return make_request("post", Route.pretraining + "/deleteCorpus", data)


def start_alignment(alignment_name: str, qa_set: str, pretrained_model: str) -> None:
    """
    Start alignment of a model

    Args:
        alignment_name (str): The name of the alignment job
        qa_set (str): The name of the QA set to use
        pretrained_model (str): The name of the pretrained model to use
    """

    data = {"alignment_name": alignment_name, "qa_set_name": qa_set, "pretrained_model": pretrained_model}

    return make_request("post", Route.alignment + "/startAlignment", data)


def upload_alignment(alignment_name: str, alignment_id: str, qa_set_id: str, pretraining_id: str) -> None:
    data = {
        "alignment_name": alignment_name,
        "alignment_id": alignment_id,
        "qa_set_id": qa_set_id,
        "pretraining_id": pretraining_id,
    }
    return make_request("post", Route.alignment + "/uploadAlignment", data)


def start_retriever_training(name: str, context: str):
    data = {"name": name, "context": context}
    make_request("post", Route.train_model, data)
    org = get_current_org()
    status_url = f"{config.ARCEE_APP_URL}/{org}/models/{name}/training"
    print(
        f'Retriever model training started - view model status at {status_url} or with arcee.get_retriever_status("{name}")'
    )


def get_retriever_status(id_or_name: str) -> Dict[str, str]:
    """Gets the status of a retriever training job"""
    return check_model_status(id_or_name)


def start_deployment(
    deployment_name: str,
    alignment: Optional[str] = None,
    retriever: Optional[str] = None,
    target_instance: Optional[str] = None,
    openai_compatability: Optional[bool] = False,
):
    data = {
        "deployment_name": deployment_name,
        "alignment_name": alignment,
        "retriever_name": retriever,
        "target_instance": target_instance,
        "openai_compatability": openai_compatability,
    }
    return make_request("post", Route.deployment + "/startDeployment", data)


def stop_deployment(deployment_name: str):
    data = {"deployment_name": deployment_name}
    return make_request("post", Route.deployment + "/stopDeployment", data)


def generate(deployment_name: str, query: str):
    data = {"deployment_name": deployment_name, "query": query}
    return make_request("post", Route.deployment + "/generate", data)


def retrieve(deployment_name: str, query: str, size: Optional[int] = 5):
    data = {"deployment_name": deployment_name, "query": query, "size": size}
    return make_request("post", Route.deployment + "/retrieve", data)


def embed(deployment_name: str, query: str):
    data = {"deployment_name": deployment_name, "query": query}
    return make_request("post", Route.deployment + "/embed", data)


def get_current_org() -> str:
    return make_request("get", Route.identity)["org"]
