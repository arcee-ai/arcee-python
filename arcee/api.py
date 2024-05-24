from typing import Dict, List, Optional, Union

from arcee import config
from arcee.api_handler import make_request
from arcee.dalm import DALM, check_model_status
from arcee.schemas.routes import Route
import csv


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

import os
import csv
import json
import arcee
from tqdm import tqdm
from requests.exceptions import ConnectionError
from datasets import load_dataset
from typing import List, Dict

def chunk_list(lst, chunk_size):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def upload_qa_pairs_with_retry(qa_set, qa_pairs, retries=3, delay=5):
    """Upload QA pairs with retry logic."""
    for attempt in range(retries):
        try:
            arcee.upload_qa_pairs(qa_set=qa_set, qa_pairs=qa_pairs)
            return True
        except ConnectionError as e:
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
        except requests.HTTPError as e:
            raise

def load_data(source):
    """Load data from CSV, JSON, or JSONL file."""
    if source.endswith('.csv'):
        with open(source, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            return list(reader)
    elif source.endswith('.json') or source.endswith('.jsonl'):
        with open(source, 'r', encoding='utf-8') as jsonfile:
            if source.endswith('.json'):
                return json.load(jsonfile)
            else:
                return [json.loads(line) for line in jsonfile]
    else:
        raise ValueError("Unsupported file format")

def validate_and_standardize_columns(data):
    """Validate dataset columns and standardize relevant data."""
    required_columns_sets = [
        {'prompt', 'messages'},
        {'question', 'answer'},
        {'prompt', 'completion'},
        {'instruction', 'response'}
    ]

    standardized_qa_pairs = []

    for row in data:
        for required_columns in required_columns_sets:
            if required_columns.issubset(row.keys()):
                # Standardize the columns to 'question' and 'answer'
                standardized_row = {}
                if 'prompt' in required_columns and ('completion' in required_columns or 'messages' in required_columns):
                    standardized_row['question'] = row['prompt']
                    standardized_row['answer'] = row.get('completion', row.get('messages', ''))
                elif 'question' in required_columns and 'answer' in required_columns:
                    standardized_row['question'] = row['question']
                    standardized_row['answer'] = row['answer']
                elif 'instruction' in required_columns and 'response' in required_columns:
                    standardized_row['question'] = row['instruction']
                    standardized_row['answer'] = row['response']

                # Add 'split' if it exists
                if 'split' in row:
                    if row['split'] in {'train', 'evaluation'}:
                        standardized_row['split'] = row['split']
                    else:
                        raise ValueError("The 'split' column must contain either 'train' or 'evaluation' as a value.")

                standardized_qa_pairs.append(standardized_row)
                break
        else:
            raise ValueError("Dataset does not contain the required columns.")
    
    return standardized_qa_pairs

def upload_instructions_to_arcee(source, qa_set, batch_size=1980, hf_token=None):
    # Determine the source type and read the QA pairs
    if os.path.exists(source):
        # Local CSV, JSON, or JSONL file path
        data = load_data(source)
    else:
        # Assume it's a Hugging Face dataset identifier
        dataset = load_dataset(source, token=hf_token)['train']
        data = dataset.to_pandas().to_dict(orient='records')  # Convert to list of dicts

    # Validate and standardize columns
    qa_pairs = validate_and_standardize_columns(data)

    # Upload dataset
    for chunk in tqdm(chunk_list(qa_pairs, batch_size), total=len(qa_pairs)//batch_size + 1):
        upload_qa_pairs_with_retry(qa_set=qa_set, qa_pairs=chunk)


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

def start_merging(
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
