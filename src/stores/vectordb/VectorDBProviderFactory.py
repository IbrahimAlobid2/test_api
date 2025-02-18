from .providers import QdrantDBProvider
from .providers import ChromaDBProvider
from .VectorDBEnums import VectorDBEnums
from controllers.BaseController import BaseController

class VectorDBProviderFactory:
    def __init__(self, config):
        self.config = config
        self.base_controller = BaseController()

    def create(self, provider: str):
        if provider == VectorDBEnums.QDRANT.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)

            return QdrantDBProvider(
                db_path=db_path,
            )
        elif provider == VectorDBEnums.CHROMA.value:
            db_path = self.base_controller.get_database_path(db_name=self.config.VECTOR_DB_PATH)

            return ChromaDBProvider(
                db_path=db_path,
            )       
        return None
