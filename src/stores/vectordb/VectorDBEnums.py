from enum import Enum
from pydantic import BaseModel

class VectorDBEnums(Enum):
    QDRANT = "QDRANT"
    CHROMA = "CHROMA"


class RetrievedDocument(BaseModel):
    text: str
    score: float