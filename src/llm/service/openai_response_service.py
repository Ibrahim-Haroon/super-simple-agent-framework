import requests
from typing import override, Optional, List, Dict, Any
from src.util.env import Env
from src.llm.memory.llm_message import LlmMessage
from src.llm.service.response_service import LlmResponseService


class OpenAILlmResponseService(LlmResponseService):
    def __init__(self, model: str = "gpt-4o"):
        self.__model = model
        self.__url = "https://api.openai.com/v1/chat/completions"
        self.__api_key = Env()["OPENAI_API_KEY"]
        self.__headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.__api_key}",
        }

    @property
    def model(self) -> str:
        return self.__model

    @override
    def response(
            self,
            role: str | None = None,
            prompt: str | None = None,
            tools: Optional[list[dict[str, Any]]] = None,
            conversation_history: Optional[List[LlmMessage]] = None,
            timeout: Optional[int] = None
    ) -> dict:
        payload = {
            "model": self.__model,
            "messages": [
                *([{
                    "role": "system",
                    "content": role
                }] if role else []),
                *[m.to_dict() for m in (conversation_history or [])],
                *([{
                    "role": "user",
                    "content": prompt
                }] if prompt else [])
            ],
            **({"tools": tools} if tools else {})
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

        return response.json()
