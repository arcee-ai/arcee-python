from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator
from strenum import StrEnum

from arcee.api_handler import make_request
from arcee.schemas.routes import Route


def check_model_status(name: str) -> Dict[str, str]:
    """
        Checks the status of a model.

        Args:
            name (str): The name of the model.

        Returns:
            Dict[str, str]: A dictionary containing the status of the model.

        Examples:
            >>> check_model_status("model_1")
            {'status': 'running'}
        """

    route = Route.train_model_status.value.format(id_or_name=name) + "?allow_demo=True"
    return make_request("get", route)


class FilterType(StrEnum):
    """
    An enumeration class representing different filter types.

    Attributes:
        fuzzy_search (str): Represents the fuzzy search filter type.
        strict_search (str): Represents the strict search filter type.
    """
    fuzzy_search = "fuzzy_search"
    strict_search = "strict_search"


class DALMFilter(BaseModel):
    """Filters available for a dalm retrieve/generation query

    Arguments:
        field_name: The field to filter on. Can be 'document' or 'name' to filter on your document's raw text or title
            Any other field will be presumed to be a metadata field you included when uploading your context data
        filter_type: Currently 'fuzzy_search' and 'strict_search' are supported. More to come soon!
            'fuzzy_search' means a fuzzy search on the provided field will be performed. The exact strict doesn't
            need to exist in the document for this to find a match. Very useful for scanning a document for some
            keyword terms
            'strict_search' means that the exact string must appear in the provided field. This is NOT an exact eq
            filter. ie a document with content "the happy dog crossed the street" will match on a strict_search of "dog"
            but won't match on "the dog". Python equivalent of `return search_string in full_string`
        value: The actual value to search for in the context data/metadata
    """

    field_name: str
    filter_type: FilterType
    value: str
    _is_metadata: bool = False

    @model_validator(mode="after")
    def set_meta(self) -> "DALMFilter":
        """document and name are reserved arcee keys. Anything else is metadata"""
        self._is_metadata = self.field_name not in ["document", "name"]
        return self


class DALM:
    """
    A class representing a DALM (Deep Active Learning Model).

    Args:
        name (str): The name of the DALM model.

    Raises:
        Exception: If the DALM model is not ready.

    Methods:
        invoke(invocation_type, query, size, filters): Invokes the DALM model with the specified invocation type, query, size, and filters.
        retrieve(query, size=3, filters=None): Retrieves contexts using the retriever for the given query.
        generate(query, size=3, filters=None): Generates a response using the generator for the given query.
    """

    def __init__(self, name: str) -> None:
        self.name = name

        retriever_api_response = check_model_status(name)

        self.model_id = retriever_api_response["id"]
        self.status = retriever_api_response["status"]

        if self.status != "training_complete":
            raise Exception("DALM model is not ready. Please wait for training to complete.")

    def invoke(
        self, invocation_type: Literal["retrieve", "generate"], query: str, size: int, filters: List[Dict]
    ) -> Dict[str, Any]:
        """
        Invokes the DALM model with the specified invocation type, query, size, and filters.

        Args:
            invocation_type (Literal["retrieve", "generate"]): The type of invocation to perform.
            query (str): The question to submit to the model.
            size (int): The maximum number of context results to retrieve or generate.
            filters (List[Dict]): Optional filters to include with the query.

        Returns:
            Dict[str, Any]: A dictionary containing the response from the DALM model.
        """

        route = Route.retrieve if invocation_type == "retrieve" else Route.generate
        payload = {"model_id": self.model_id, "query": query, "size": size, "filters": filters, "id": self.model_id}
        return make_request("post", route, body=payload)

    def retrieve(self, query: str, size: int = 3, filters: Optional[List[DALMFilter]] = None) -> Dict:
        """
        Retrieves contexts using the retriever for the given query.

        Args:
            query (str): The question to submit to the model.
            size (int, optional): The maximum number of context results to retrieve. Defaults to 3.
            filters (Optional[List[DALMFilter]], optional): Optional filters to include with the query. Defaults to None.

        Returns:
            Dict: A dictionary containing the retrieved contexts.

        Example:
            >>> retrieve("What is the capital of France?")
            {'contexts': ['Paris is the capital of France.', 'The capital of France is Paris.']}
        """

        filters = filters or []
        ret_filters = [DALMFilter.model_validate(f).model_dump() for f in filters]
        return self.invoke("retrieve", query, size, ret_filters)

    def generate(self, query: str, size: int = 3, filters: Optional[List[DALMFilter]] = None) -> Dict:
        """
        Generates a response using the generator for the given query.

        Args:
            query (str): The question to submit to the model.
            size (int, optional): The maximum number of context results to retrieve. Defaults to 3.
            filters (Optional[List[DALMFilter]], optional): Optional filters to include with the query. Defaults to None.

        Returns:
            Dict: A dictionary containing the generated response.

        Example:
            >>> generate("What is the capital of France?")
            {'response': 'The capital of France is Paris.'}
        """

        filters = filters or []
        gen_filters = [DALMFilter.model_validate(f).model_dump() for f in filters]
        return self.invoke("generate", query, size, gen_filters)
