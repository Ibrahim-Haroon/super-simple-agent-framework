import json
import random
import uuid
from textwrap import dedent
from typing import Any, Dict, Callable, Optional, List
from src.agent.agent import Agent
from src.llm.memory.conversation_memory import ConversationMemory
from src.llm.memory.llm_message import LlmMessage
from src.llm.service.openai_response_service import OpenAILlmResponseService
from src.llm.service.response_service import LlmResponseService
from src.llm.service.openai_embedding_service import OpenAILlmEmbeddingService
from src.vector_db.redis_vector_db import RedisVectorDB
from src.vector_db.vector_db import VectorDB
from src.vector_db.pg_vector_db import PGVectorDB
from src.tools.decorator import get_tool_definitions, tool


KNOWLEDGE_BASE: Dict[str, str] = {
    "kb-001": "To reset your password, visit account.example.com/reset and enter your registered email address. You will receive a reset link within 5 minutes. Links expire after 30 minutes.",
    "kb-002": "Billing cycles run on the 1st of each month. Invoices are sent via email and are also available under Settings → Billing. We accept Visa, Mastercard, and ACH transfers.",
    "kb-003": "The API rate limit is 1,000 requests per minute on the Starter plan and 10,000 on Pro. Exceeded requests return HTTP 429. Contact support to request a temporary limit increase.",
    "kb-004": "To export your data, go to Settings → Data → Export. Exports are generated as a ZIP containing JSON files and are emailed to your account address within 10 minutes.",
    "kb-005": "Two-factor authentication (2FA) can be enabled under Settings → Security. We support authenticator apps (TOTP) and SMS. Hardware keys (WebAuthn) are available on the Enterprise plan.",
    "kb-006": "Integrations with Slack, Jira, and GitHub are available under Settings → Integrations. Each integration requires an OAuth connection and can be scoped to specific projects.",
    "kb-007": "Our SLA guarantees 99.9% uptime on the Pro plan and 99.99% on Enterprise. Scheduled maintenance windows are announced 48 hours in advance via the status page and email.",
    "kb-008": "To add a team member, go to Settings → Team → Invite. Free plans support up to 3 seats; Starter up to 10; Pro and Enterprise have unlimited seats.",
    "kb-009": "Webhooks can be configured under Settings → Developers → Webhooks. We send POST requests with an HMAC-SHA256 signature header for payload verification.",
    "kb-010": "To cancel your subscription, go to Settings → Billing → Cancel. Your access continues until the end of the current billing period. Data is retained for 30 days post-cancellation.",
}


class TechSupportAgent(Agent):
    def __init__(
            self,
            llm_service: LlmResponseService,
            vector_db: VectorDB,
            memory: ConversationMemory,
            tools: Optional[List[Callable]] = None
    ):
        self.__llm_service = llm_service
        self.__vector_db = vector_db
        self.__memory = memory
        self.__conversation_id = uuid.uuid4()

        self.__tools = get_tool_definitions(*tools)
        self._tool_map: Dict[str, Callable] = {t.__name__: t for t in tools}

    @property
    def agent_type(self) -> str:
        return "Tech Support Agent"

    @property
    def role(self) -> str:
        return dedent(
            """
            You are a helpful technical support agent for Example Corp.

            You will be given a set of KNOWLEDGE BASE ARTICLES that are
            relevant to the user's question. Use them as your primary source
            of truth. If the articles do not contain enough information to
            answer confidently, say so and offer to escalate the ticket.

            Rules:
            - Be concise and friendly.
            - Always cite which KB article(s) you used (e.g. "[kb-003]").
            - Never fabricate steps or policy details not present in the articles.
            - If the user is frustrated or the issue is unresolved, offer to
              escalate by calling the escalate_ticket tool.
            """
        ).strip()

    def execute(self, task: str) -> str:
        relevant_docs = self.__vector_db.similarity_search(task)
        grounding_context = self.__build_context(relevant_docs)

        grounded_prompt = f"{grounding_context}\n\nUser question: {task}"

        self.__memory.add(
            self.__conversation_id,
            LlmMessage(role="user", content=task),
        )

        response = self.__llm_service.response(
            role=self.role,
            prompt=grounded_prompt,
            tools=self.__tools,
            conversation_history=self.__memory.history(self.__conversation_id),
        )

        while response["choices"][0]["finish_reason"] == "tool_calls":
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
                            "arguments": tool_call["function"]["arguments"],
                        },
                    } for tool_call in tool_calls],
                ),
            )

            for tool_call in tool_calls:
                tool_name = tool_call["function"]["name"]
                tool_args = json.loads(tool_call["function"]["arguments"])
                tool_result = (
                    self._tool_map[tool_name](**tool_args)
                    if tool_name in self._tool_map
                    else {"error": f"Unknown tool: {tool_name}"}
                )

                self.__memory.add(
                    self.__conversation_id,
                    LlmMessage(
                        role="tool",
                        content=str(tool_result),
                        tool_call_id=tool_call["id"],
                    ),
                )

            response = self.__llm_service.response(
                role=self.role,
                prompt=None,
                tools=self.__tools or None,
                conversation_history=self.__memory.history(self.__conversation_id),
            )

        assistant_message = response["choices"][0]["message"]["content"]
        self.__memory.add(
            self.__conversation_id,
            LlmMessage(role="assistant", content=assistant_message),
        )

        return assistant_message

    def reset_conversation(self) -> None:
        self.__conversation_id = uuid.uuid4()

    def get_conversation_history(self) -> List[LlmMessage]:
        return self.__memory.history(self.__conversation_id)

    @staticmethod
    def __build_context(docs: List[Any]) -> str:
        if not docs:
            return "KNOWLEDGE BASE ARTICLES:\nNo relevant articles found."

        lines = ["KNOWLEDGE BASE ARTICLES:"]
        for doc in docs:
            doc_id = doc.get("id", "unknown")
            text = doc.get("text", "")
            score = doc.get("score")
            score_str = f"  (relevance: {float(score):.2f})" if score is not None else ""
            lines.append(f"[{doc_id}]{score_str}: {text}")

        return "\n".join(lines)


@tool
def check_service_status(service_name: str) -> Dict[str, Any]:
    """
    Check the current operational status of a named service or feature.

    :param service_name: The name of the service to check (e.g. 'API', 'Billing', 'Webhooks')
    """
    statuses = [
        {
            "service": service_name,
            "status": "operational",
            "last_incident": None,
            "message": f"{service_name} is fully operational with no known issues.",
        },
        {
            "service": service_name,
            "status": "degraded_performance",
            "last_incident": "2024-06-01T14:30:00Z",
            "message": f"{service_name} is experiencing degraded performance. Our team is investigating.",
        },
        {
            "service": service_name,
            "status": "partial_outage",
            "last_incident": "2024-06-01T13:00:00Z",
            "message": f"{service_name} is currently experiencing a partial outage."
        },
        {
            "service": service_name,
            "status": "major_outage",
            "last_incident": "2024-06-01T12:00:00Z",
            "message": f"{service_name} is currently experiencing a major outage. All users are likely affected."
        }
    ]

    return random.choice(statuses)


def seed_knowledge_base(vector_db: VectorDB) -> None:
    """Embed and store all KB articles in the vector DB (idempotent)."""
    print("Seeding knowledge base into vector DB...")
    vector_db.add_documents(
        document_ids=list(KNOWLEDGE_BASE.keys()),
        documents=list(KNOWLEDGE_BASE.values()),
    )
    print(f"  {len(KNOWLEDGE_BASE)} articles indexed.\n")


def main():
    embedding_service = OpenAILlmEmbeddingService(dimensions=300)

    """
    PGVectorDB(
        embedding_service=embedding_service,
        host="localhost",
        port=5432,
        database="postgres",
        user="postgres",
        password="",
        table_name="support_kb",
    )
    """
    vector_db = RedisVectorDB(
        embedding_service=embedding_service,
        host="localhost",
        port=6379,
        index_name="test1",
        vector_dim=300
    )

    seed_knowledge_base(vector_db)

    agent = TechSupportAgent(
        llm_service=OpenAILlmResponseService(),
        vector_db=vector_db,
        memory=ConversationMemory(),
        tools=[
            check_service_status
        ]
    )

    print("=" * 60)
    print("🛠️  Example Corp — Technical Support")
    print("=" * 60)
    print("Ask me anything about your account, billing, API, or integrations.")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"quit", "exit", "q"}:
                print("\nThanks for contacting support. Goodbye!\n")
                break

            response = agent.execute(user_input)
            print(f"\nAgent: {response}\n")

        except KeyboardInterrupt:
            print("\n\nGoodbye!\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
