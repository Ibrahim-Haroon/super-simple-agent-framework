from abc import ABC, abstractmethod
from typing import Optional, List, Any
from src.llm.memory.llm_message import LlmMessage


class LlmResponseService(ABC):
    @abstractmethod
    def response(
            self,
            role: Optional[str],
            prompt: Optional[str],
            tools: Optional[list[dict[str, Any]]],
            conversation_history: Optional[List[LlmMessage]],
            timeout: Optional[int] = None
    ) -> dict:
        """
        This method is used as a baseline behavior for all LLM Response Services

        :param role: The behavior/persona for the model to inherit
        :param prompt: The task for the model to complete
        :param tools: Tools available to the model during response generation
        :param conversation_history: The conversation history to provide context for the model
        :param timeout: Amount of seconds to wait before throwing requests.exceptions.Timeout
        :return: response object from LLM
        :rtype: dict
        :exception: requests.exceptions.Timeout and requests.exceptions.RequestException
        """
        pass
