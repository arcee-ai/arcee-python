__version__ = "0.0.14"


from arcee.api import get_dalm, get_dalm_status, train_dalm, upload_doc, upload_docs
from arcee.config import ARCEE_API_KEY, ARCEE_APP_URL
from arcee.dalm import DALM

if not ARCEE_API_KEY:
    raise Exception(f"ARCEE_API_KEY must be in the environment. You can retrieve your API key from {ARCEE_APP_URL}")


__all__ = ["upload_docs", "upload_doc", "train_dalm", "get_dalm", "DALM", "get_dalm_status"]
