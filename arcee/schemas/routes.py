from strenum import StrEnum


class Route(StrEnum):
    contexts = "contexts"
    train_model = "models/train"
    train_model_status = "models/status/{id_or_name}?allow_demo=True"
    train_retriever = "retrievers/train"
    train_retriever_status = "retrievers/status/{id_or_name}"
    identity = "whoami"
    retrieve = "models/retrieve"
    generate = "models/generate"
    pretraining = "pretraining"
    alignment = "alignment"
    deployment = "deployment"