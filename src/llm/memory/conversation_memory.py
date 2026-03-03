import uuid
from typing import List, Dict
from collections import defaultdict
from src.llm.memory.llm_message import LlmMessage


class ConversationMemory:
    def __init__(self):
        self.__store: Dict[uuid.UUID, List[LlmMessage]] = defaultdict(list)

    def history(self, conversation_id: uuid.UUID) -> List[LlmMessage]:
        return self.__store[conversation_id].copy()

    def add(self, conversation_id: uuid.UUID, message: LlmMessage) -> None:
        self.__store[conversation_id].append(message)
