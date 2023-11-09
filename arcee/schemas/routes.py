from strenum import StrEnum


class Route(StrEnum):
    """
    An enumeration class representing different API routes.

    Attributes:
        contexts (str): The route for retrieving contexts.
        train_model (str): The route for training a model.
        train_model_status (str): The route for checking the status of a model.
        train_retriever (str): The route for training a retriever.
        train_retriever_status (str): The route for checking the status of a retriever.
        identity (str): The route for retrieving identity information.
        retrieve (str): The route for retrieving data from a model.
        generate (str): The route for generating data from a model.
    """

    contexts = "contexts"
    train_model = "models/train"
    train_model_status = "models/status/{id_or_name}"
    train_retriever = "retrievers/train"
    train_retriever_status = "retrievers/status/{id_or_name}"
    identity = "whoami"
    retrieve = "models/retrieve"
    generate = "models/generate"
