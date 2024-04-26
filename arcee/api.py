from typing import Dict, List, Optional, Union

from arcee import config
from arcee.api_handler import make_request
from arcee.dalm import DALM, check_model_status
from arcee.schemas.routes import Route

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

    return make_request("post", Route.pretraining+"/corpusUpload", data)

def upload_qa_pairs(qa_set: str, qa_pairs: List[Dict[str, str]]) -> Dict[str, str]:
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
        if "question" not in qa or "answer" not in qa:
            raise Exception("Each QA pair must have a 'question' and an 'answer' key")

        qa_list.append({"question": qa["question"], "answer": qa["answer"]})

    data = {"qa_set_name": qa_set, "qa_pairs": qa_list}
    return make_request("post", Route.alignment+"/qaUpload", data)

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

    return make_request("post", Route.pretraining+"/startTraining", data)

def delete_corpus(corpus: str) -> None:
    """
    Delete a corpus

    Args:
        corpus (str): The name of the corpus to delete
    """

    data = {"corpus_name": corpus}

    return make_request("post", Route.pretraining+"/deleteCorpus", data)

def start_alignment(alignment_name: str, qa_set: str, pretrained_model: str) -> None:
    """
    Start alignment of a model

    Args:
        alignment_name (str): The name of the alignment job
        qa_set (str): The name of the QA set to use
        pretrained_model (str): The name of the pretrained model to use
    """

    data = {"alignment_name": alignment_name, "qa_set_name": qa_set, "pretrained_model": pretrained_model}

    return make_request("post", Route.alignment+"/startAlignment", data)

def upload_alignment(alignment_name: str, alignment_id: str, qa_set_id: str, pretraining_id: str) -> None:
    data = {"alignment_name": alignment_name, "alignment_id": alignment_id, "qa_set_id": qa_set_id, "pretraining_id": pretraining_id}
    return make_request("post", Route.alignment+"/uploadAlignment", data)

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

def start_deployment(deployment_name: str, alignment: Optional[str] = None, retriever: Optional[str] = None):
    data = {"deployment_name": deployment_name, "alignment_name": alignment, "retriever_name": retriever}
    return make_request("post", Route.deployment+"/startDeployment", data)

def stop_deployment(deployment_name: str):
    data = {"deployment_name": deployment_name}
    return make_request("post", Route.deployment+"/stopDeployment", data)

def generate(deployment_name: str, query: str):
    data = {"deployment_name": deployment_name, "query": query}
    return make_request("post", Route.deployment+"/generate", data)

def retrieve(deployment_name: str, query: str, size: Optional[int] = 5):
    data = {"deployment_name": deployment_name, "query": query, "size": size}
    return make_request("post", Route.deployment+"/retrieve", data)

def embed(deployment_name: str, query: str):
    data = {"deployment_name": deployment_name, "query": query}
    return make_request("post", Route.deployment+"/embed", data)

def get_current_org() -> str:
    return make_request("get", Route.identity)["org"]