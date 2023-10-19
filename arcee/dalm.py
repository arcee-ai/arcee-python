from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, model_validator
from strenum import StrEnum

from arcee.api_handler import make_request
from arcee.schemas.routes import Route


def check_model_status(name: str) -> Dict[str, str]:
    route = Route.train_model_status.value.format(id_or_name=name)
    return make_request("get", route)


class FilterType(StrEnum):
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
        route = Route.retrieve if invocation_type == "retrieve" else Route.generate
        payload = {"model_id": self.model_id, "query": query, "size": size, "filters": filters, "id": self.model_id}
        return make_request("post", route, body=payload)

    def retrieve(self, query: str, size: int = 3, filters: Optional[List[DALMFilter]] = None) -> Dict:
        """Retrieve {size} contexts with your retriever for the given query

        Arguments:
            query: The question to submit to the model
            size: The max number of context results to retrieve (can be less if filters are provided)
            filters: Optional filters to include with the query. This will restrict which context data the model can
                retrieve from the context dataset
        """
        filters = filters or []
        ret_filters = [DALMFilter.model_validate(f).model_dump() for f in filters]
        return self.invoke("retrieve", query, size, ret_filters)

    def generate(self, query: str, size: int = 3, filters: Optional[List[DALMFilter]] = None) -> Dict:
        """Generate a response using {size} contexts with your generator for the given query

        Arguments:
            query: The question to submit to the model
            size: The max number of context results to retrieve (can be less if filters are provided)
            filters: Optional filters to include with the query. This will restrict which context data the model can
                retrieve from the context dataset
        """
        filters = filters or []
        gen_filters = [DALMFilter.model_validate(f).model_dump() for f in filters]
        return self.invoke("generate", query, size, gen_filters)
