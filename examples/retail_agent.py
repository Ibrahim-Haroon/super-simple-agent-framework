import json
import uuid
from textwrap import dedent
from typing import Any, Dict, Callable, Optional, List
from src.agent.agent import Agent
from src.llm.memory.conversation_memory import ConversationMemory
from src.llm.memory.llm_message import LlmMessage
from src.llm.service.openai_response_service import OpenAILlmResponseService
from src.llm.service.response_service import LlmResponseService
from src.tools.tool_decorator import get_tool_definitions, tool


class RetailAgent(Agent):
    def __init__(
            self,
            llm_service: LlmResponseService,
            memory: ConversationMemory,
            tools: Optional[List[Callable]] = None
    ):
        self.__llm_service = llm_service
        self.__memory = memory
        self.__conversation_id = uuid.uuid4()

        self.__tools = get_tool_definitions(*tools)
        self._tool_map: Dict[str, Callable] = {
            t.__name__: tool for t in tools
        }

    @property
    def role(self) -> str:
        return dedent(
            """
            You are a customer service agent for ...
            """
        ).strip()

    @property
    def agent_type(self) -> str:
        return "Retail Agent"

    def execute(self, task: str) -> str:
        self.__memory.add(
            self.__conversation_id,
            LlmMessage(role="user", content=task)
        )

        response = self.__llm_service.response(
            role=self.role,
            prompt=task,
            tools=self.__tools,
            conversation_history=self.__memory.history(self.__conversation_id)
        )

        if response["choices"][0]["finish_reason"] == "tool_calls":
            tool_calls = response["choices"][0]["message"]["tool_calls"]

            self.__memory.add(
                self.__conversation_id,
                LlmMessage(
                    role="assistant",
                    content=response["choices"][0]["message"].get("content"),
                    tool_calls=[{
                        "id": tool_call["id"],
                        "type": tool_call["type"],
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
                    } for tool_call in tool_calls]
                )
            )

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])

                if tool_name in self._tool_map:
                    tool_result = self._tool_map[tool_name](**tool_args)

                    self.__memory.add(
                        self.__conversation_id,
                        LlmMessage(
                            role="tool",
                            content=str(tool_result),
                            tool_call_id=tool_call["id"]
                        )
                    )

            final_response = self.__llm_service.response(
                role=self.role,
                prompt=None,  # Empty prompt, context is in conversation history
                tools=self.__tools,
                conversation_history=self.__memory.history(self.__conversation_id)
            )

            assistant_message = final_response["choices"][0]["message"]["content"]
        else:
            assistant_message = response["choices"][0]["message"]["content"]

        self.__memory.add(
            self.__conversation_id,
            LlmMessage(role="assistant", content=assistant_message)
        )

        return assistant_message

    def get_conversation_history(self):
        return self.__memory.history(self.__conversation_id)

    def reset_conversation(self):
        self.__conversation_id = uuid.uuid4()


@tool
def get_order_status(order_id: str) -> Dict[str, Any]:
    pass


@tool
def search_products(query: str) -> List[Dict[str, Any]]:
    pass


def main():
    agent = RetailAgent(
        OpenAILlmResponseService(),
        ConversationMemory(),
        tools=[
            get_order_status,
            search_products
        ]
    )

    print("=" * 60)
    print("🏔️  Welcome to Customer Service! 🏔️")
    print("=" * 60)
    print("I'm here to help you with:")
    print("  • Order status and tracking")
    print("  • Product recommendations")
    print("\nType 'quit' or 'exit' to end the conversation")
    print("=" * 60)
    print()

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n️ Thanks for visiting us! \n")
                break

            response = agent.execute(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\n️ Thanks for visiting us ️\n")
            break
        except Exception as e:
            print(f"\n❌ Oops! Something went wrong: {e}\n")
            print("Let's try that again!\n")


if __name__ == "__main__":
    main()
