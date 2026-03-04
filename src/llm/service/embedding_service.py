from abc import ABC, abstractmethod
from typing import Optional, List


class LlmEmbeddingService(ABC):
    def __init__(self, dimensions: int):
        self._dimensions = dimensions

    @abstractmethod
    def embed(
            self,
            text: str,
            timeout: Optional[int] = None
    ) -> List[float]:
        """
        Generates a vector embedding for a single text input.

        :param text: The text to embed
        :param timeout: Amount of seconds to wait before throwing requests.exceptions.Timeout
        :return: Embedding vector as a list of floats
        :rtype: List[float]
        :exception: requests.exceptions.Timeout and requests.exceptions.RequestException
        """
        pass

    @abstractmethod
    def embed_batch(
            self,
            texts: List[str],
            timeout: Optional[int] = None
    ) -> List[List[float]]:
        """
        Generates vector embeddings for a batch of texts.

        :param texts: List of texts to embed
        :param timeout: Amount of seconds to wait before throwing requests.exceptions.Timeout
        :return: List of embedding vectors
        :rtype: List[List[float]]
        :exception: requests.exceptions.Timeout and requests.exceptions.RequestException
        """
        pass