__version__ = "0.0.15"

import os

from arcee.api import get_dalm, get_dalm_status, train_dalm, upload_doc, upload_docs
from arcee.config import ARCEE_API_KEY
from arcee.dalm import DALM

if not ARCEE_API_KEY:
    # We check this because it's impossible the user imported arcee, _then_ set the env, then imported again
    ARCEE_API_KEY = os.getenv("ARCEE_API_KEY", "")
    while not ARCEE_API_KEY:
        ARCEE_API_KEY = input("ARCEE_API_KEY not found. Please input api key: ")
    os.environ["ARCEE_API_KEY"] = ARCEE_API_KEY

__all__ = ["upload_docs", "upload_doc", "train_dalm", "get_dalm", "DALM", "get_dalm_status"]
