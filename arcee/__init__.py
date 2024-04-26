__version__ = "1.0.6"

import os

from arcee import config
from arcee.api import upload_docs, upload_corpus_folder, upload_qa_pairs, start_alignment, start_pretraining, start_retriever_training, get_retriever_status, start_deployment, stop_deployment, generate, retrieve, delete_corpus, upload_alignment
from arcee.dalm import DALM, DALMFilter

if not config.ARCEE_API_KEY:
    # We check this because it's impossible the user imported arcee, _then_ set the env, then imported again
    config.ARCEE_API_KEY = os.getenv("ARCEE_API_KEY", "")
    while not config.ARCEE_API_KEY:
        config.ARCEE_API_KEY = input("ARCEE_API_KEY not found in environment. Please input api key: ")
    os.environ["ARCEE_API_KEY"] = config.ARCEE_API_KEY

__all__ = ["upload_docs", "DALM", "DALMFilter", "upload_corpus_folder", "upload_qa_pairs", "start_alignment", "start_pretraining", "start_retriever_training", "get_retriever_status", "start_deployment", "stop_deployment", "generate", "retrieve", "delete_corpus", "upload_alignment"]
