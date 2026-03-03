from typing import Literal, Optional, List, Dict, Any
from dataclasses import dataclass, field


@dataclass
class LlmMessage:
    """
    Represents a message that can be passed in the payload of request to LLM as conversation history
    """
    role: Literal["user", "assistant", "tool"]
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        msg_dict: Dict[str, Any] = {
            "role": self.role,
            "content": self.content
        }

        if self.role == "tool":
            msg_dict["tool_call_id"] = self.tool_call_id
        elif self.tool_calls:
            msg_dict["tool_calls"] = self.tool_calls

        return msg_dict
