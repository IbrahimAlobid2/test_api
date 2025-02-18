import logging
from typing import List, Dict, Any

from .BaseController import BaseController
from .ProcessController import ProcessController
from stores.vectordb.VectorDBProviderFactory import VectorDBProviderFactory
from stores.llm.LLMProviderFactory import LLMProviderFactory


logger = logging.getLogger(__name__)

class RAGController(BaseController):
    """
    A controller class that handles RAG (Retrieval-Augmented Generation) workflows:
    - Setting up LLM and VectorDB clients
    - Indexing data into the vector database
    - Searching the vector database via similarity search
    """

    def __init__(self ,em :bool =False):
        """
        Initialize RAGController by creating LLM and VectorDB clients.
        Assumes `BaseController` initializes `self.app_settings` with the necessary configs.
        """
        super().__init__()
        self.em =em
        # LLM Provider
        self.llm_provider_factory = LLMProviderFactory(self.app_settings ,azure = False)
        self.text_embedding_client = self.llm_provider_factory.create(
            provider=self.app_settings.EMBEDDING_BACKEND 
        )
        self.text_embedding_client.set_embedding_model(
            model_id=self.app_settings.EMBEDDING_MODEL_ID
        )

        # VectorDB Provider
        self.vectordb_provider_factory = VectorDBProviderFactory(self.app_settings)
        self.vectordb_client = self.vectordb_provider_factory.create(
            provider=self.app_settings.VECTOR_DB_BACKEND
        )
        self.vectordb_client.connect()

        # Path to your dataset (CSV)
        self.data_csv = self.get_dataset_path(db_name=self.app_settings.DATASET)

    def index_into_vector_db(self) ->dict:
        """
        Reads data from a CSV file, creates a fresh vector DB collection,
        and inserts vectors for each document.

        :return: True if indexing is successful, False otherwise.
        """
        
        if not self.em :
            vectordb_info = self.vectordb_client.get_collection_info(collection_name=self.app_settings.COLLECTION_NAME)
            return vectordb_info
        else:
            try:
                process_controller = ProcessController()
                df, file_name = process_controller.get_file_loader(self.data_csv)
                docs, metadatas, ids, embeddings = process_controller.prepare_data_for_injection(
                    df, file_name 
                )

                # Create or reset the collection
                self.vectordb_client.create_collection(
                    collection_name=self.app_settings.COLLECTION_NAME,
                    do_reset=True
                )

                # Insert documents
                self.vectordb_client.insert_many(
                    collection_name=self.app_settings.COLLECTION_NAME,
                    texts=docs,
                    metadata=metadatas,
                    vectors=embeddings,
                    record_ids=ids,
                )

                logger.info("Data is stored in the vector database.")
                vectordb_info = self.vectordb_client.get_collection_info(
                    collection_name=self.app_settings.COLLECTION_NAME
                )
                logger.info(f"Vector DB Info: {vectordb_info}")

                return vectordb_info

            except Exception as e:
                logger.error(f"Error during vector DB indexing: {e}", exc_info=True)
                return None

    def search_vector_db_collection(self, text: str, limit: int = 3) -> List[Dict[str, Any]]:
        """
        Embeds the input text, then performs a similarity search in the vector database.

        :param text: The query text to embed.
        :param limit: The maximum number of matching documents to return.
        :return: A list of similarity search results (or an empty list if none are found or an error occurs).
        """
        try:
            # Embed the text
            vector = self.text_embedding_client.embed_text(text=text)
            if not vector:
                logger.warning("Embedding returned an empty vector; returning empty result set.")
                return []

            # Perform similarity search in the Vector DB
            results = self.vectordb_client.search_by_vector(
                collection_name=self.app_settings.COLLECTION_NAME,
                vector=vector,
                limit=limit
            )
            if not results:
                return []
            return results

        except Exception as e:
            logger.error(f"Error during vector DB search: {e}", exc_info=True)
            return []
