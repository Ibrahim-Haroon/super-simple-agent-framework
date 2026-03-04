from abc import ABC, abstractmethod
from typing import Optional, Any
from src.llm.service.embedding_service import LlmEmbeddingService


class VectorDB(ABC):
    def __init__(self, embedding_service: LlmEmbeddingService):
        self._embedding_service = embedding_service

    @abstractmethod
    def add_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        """
        Add documents to the VectorDB

        :param document_ids: The ids to assign to the documents
        :param documents: The documents to add to the VectorDB
        :return: None
        """
        pass

    @abstractmethod
    def update_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        """
        Update documents in the VectorDB

        :param document_ids: The ids of the documents to update in the VectorDB
        :param documents: The updated documents to replace the existing documents in the VectorDB
        :return: None
        """
        pass

    @abstractmethod
    def delete_documents(
            self,
            document_ids: list[str],
    ) -> None:
        """
        Delete documents from the VectorDB

        :param document_ids: The ids of the documents to delete from the VectorDB
        :return: None
        """
        pass

    @abstractmethod
    def similarity_search(
            self,
            query: str,
            top_k: Optional[int] = 5,
    ) -> list[Any]:
        """
        Search for relevant documents in the VectorDB based on a query

        :param query: The text query to search for
        :param top_k: The number of results to return
        :return: list of relevant documents from VectorDB
        :rtype: list[Any]
        """
        pass
