__version__ = "1.3.2"

import os

from arcee import config
from arcee.api import (
    corpus_status,
    delete_corpus,
    deployment_status,
    generate,
    get_retriever_status,
    list_pretrainings,
    mergekit_evolve,
    mergekit_yaml,
    merging_status,
    retrieve,
    start_alignment,
    start_deployment,
    start_pretraining,
    start_retriever_training,
    stop_deployment,
    upload_alignment,
    upload_corpus_folder,
    upload_docs,
    upload_hugging_face_dataset_qa_pairs,
    upload_qa_pairs,
    upload_qa_pairs_from_csv,
)
from arcee.dalm import DALM, DALMFilter

if not config.ARCEE_API_KEY:
    # We check this because it's impossible the user imported arcee, _then_ set the env, then imported again
    # We don't want to block or prompt here because this will interfere with CLI usage
    config.ARCEE_API_KEY = os.getenv("ARCEE_API_KEY", "")

__all__ = [
    "upload_docs",
    "DALM",
    "DALMFilter",
    "upload_corpus_folder",
    "upload_qa_pairs",
    "start_alignment",
    "start_pretraining",
    "start_retriever_training",
    "get_retriever_status",
    "start_deployment",
    "stop_deployment",
    "generate",
    "retrieve",
    "delete_corpus",
    "corpus_status",
    "upload_alignment",
    "mergekit_evolve",
    "mergekit_yaml",
    "upload_qa_pairs_from_csv",
    "upload_hugging_face_dataset_qa_pairs",
    "list_pretrainings",
    "deployment_status",
    "merging_status",

]
