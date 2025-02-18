import uuid
import chromadb
from ..VectorDBInterface import VectorDBInterface

import logging
from typing import List
from pydantic import BaseModel
class RetrievedDocument(BaseModel):
    text: str
    score: float
    
class ChromaDBProvider(VectorDBInterface):

    def __init__(self, db_path: str):
        self.client = None
        self.db_path = db_path
        self.distance_method = "cosine"
        self.logger = logging.getLogger(__name__)

    def connect(self):
        # Create a persistent Chroma client that stores data at the given path.
        self.client = chromadb.PersistentClient(path=self.db_path)

    def disconnect(self):
        self.client = None

    def is_collection_existed(self, collection_name: str) -> bool:
        # Chroma’s list_collections() returns a list of collection objects; we assume each has a "name" attribute.
        collections = self.client.list_collections()
        return any(col.name == collection_name for col in collections)

    def list_all_collections(self) -> List:
        return self.client.list_collections()

    def get_collection_info(self, collection_name: str) -> dict:
        try:
            collection = self.client.get_collection(name=collection_name)
            info = {
                "name": collection.name,
                "count": collection.count(),
                "metadata": collection.metadata
            }
            return info
        except Exception as e:
            self.logger.error(f"Error getting collection info: {e}")
            return {}

    def delete_collection(self, collection_name: str):
        if self.is_collection_existed(collection_name):
            self.client.delete_collection(name=collection_name)


    def create_collection(self, collection_name: str, embedding_size: int =None, do_reset: bool = False):
        if do_reset:
            self.delete_collection(collection_name)
        if not self.is_collection_existed(collection_name):

            metadata = {"hnsw:space": self.distance_method}
            self.client.create_collection(name=collection_name, metadata=metadata)
            return True
        return False

    def insert_one(self, collection_name: str, text: str, vector: list,
                   metadata: dict = None, record_id: str = None):
        if not self.is_collection_existed(collection_name):
            self.logger.error(f"Cannot insert record into non-existent collection: {collection_name}")
            return False
        try:
            collection = self.client.get_collection(name=collection_name)
            if record_id is None:
                record_id = str(uuid.uuid4())
            if metadata is None:
                metadata = {}
            collection.add(
                documents=[text],
                metadatas=[metadata],
                ids=[record_id],
                embeddings=[vector]
            )
        except Exception as e:
            self.logger.error(f"Error while inserting record: {e}")
            return False
        return True

    def insert_many(self, collection_name: str, texts: list, 
                    vectors: list, metadata: list = None, 
                    record_ids: list = None, batch_size: int = 50):
        if metadata is None:
            metadata = [{}] * len(texts)
            
            
        if record_ids is None:
            record_ids = [str(i) for i in range(len(texts))]
            
        try:
            collection = self.client.get_collection(name=collection_name)
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                batch_vectors = vectors[i:i + batch_size]
                batch_metadata = metadata[i:i + batch_size]
                batch_record_ids = record_ids[i:i + batch_size]
                collection.add(
                    documents=batch_texts,
                    metadatas=batch_metadata,
                    ids=batch_record_ids,
                    embeddings=batch_vectors
                )
        except Exception as e:
            self.logger.error(f"Error while inserting batch: {e}")
            return False
        return True

    def search_by_vector(self, collection_name: str, vector: list, limit: int = 5):
        try:
            collection = self.client.get_collection(name=collection_name)

            results = collection.query(query_embeddings=[vector], n_results=limit, include=["documents", "distances"])
            if not results or not results.get("ids") or len(results["ids"][0]) == 0:
                return None

            retrieved_documents = []
            # Because we passed a single query vector, each key in results is a list containing one list.
            for doc, distance in zip(results["documents"][0], results["distances"][0]):
                retrieved_documents.append(RetrievedDocument(score=distance, text=doc))
            return retrieved_documents
        except Exception as e:
            self.logger.error(f"Error during search: {e}")
            return None
