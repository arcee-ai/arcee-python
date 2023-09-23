from enum import Enum


class Route(str, Enum):
    contexts = "contexts"
    train_model = "models/train"
    train_model_status = "models/status/{id_or_name}"
    train_retriever = "retrievers/train"
    train_retriever_status = "retrievers/status/{id_or_name}"
