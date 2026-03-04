import requests
from typing import override, Optional, List
from src.util.env import Env
from src.llm.service.embedding_service import LlmEmbeddingService


class OpenAILlmEmbeddingService(LlmEmbeddingService):
    def __init__(
            self,
            dimensions: int = 300,
            model: str = "text-embedding-3-small"
    ):
        super().__init__(dimensions)
        self.__model = model
        self.__url = "https://api.openai.com/v1/embeddings"
        self.__api_key = Env()["OPENAI_API_KEY"]
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}",
        }

    @property
    def model(self) -> str:
        return self.__model

    @override
    def embed(
            self,
            text: str,
            timeout: Optional[int] = None
    ) -> List[float]:
        payload = {
            "model": self.__model,
            "input": text,
            "dimension": self.__dimensions
        }

        try:
            response = requests.post(
                url=self.__url,
                headers=self.__headers,
                json=payload,
                timeout=timeout
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("OpenAI must be down or increase timeout duration")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"OpenAI request failed: {e}") from e

        return response.json()["data"][0]["embedding"]

    @override
    def embed_batch(
            self,
            texts: List[str],
            timeout: Optional[int] = None
    ) -> List[List[float]]:
        payload = {
            "model": self.__model,
            "input": texts,
            "dimension": self.__dimensions
        }

        try:
            response = requests.post(
                url=self.__url,
                headers=self.__headers,
                json=payload,
                timeout=timeout
            )
        except requests.exceptions.Timeout:
            raise TimeoutError("OpenAI must be down or increase timeout duration")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"OpenAI request failed: {e}") from e

        data = response.json()["data"]
        return [item["embedding"] for item in sorted(data, key=lambda x: x["index"])]
