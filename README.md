# Super Simple Agent Framework

A lightweight, minimal Python framework for building LLM-powered agents with tool use, conversation memory, and vector database retrieval. No magic, no bloat — just clean abstractions you can extend.

---

## Features

- **Modular agent design** — extend the `Agent` base class to build any agent type
- **Tool use via decorator** — `@tool` auto-generates OpenAI-compatible JSON schemas from Python type hints and docstrings
- **Conversation memory** — session-scoped message history for multi-turn interactions
- **RAG-ready** — pluggable `VectorDB` abstraction for retrieval-augmented generation
- **Provider-agnostic** — swap LLM and embedding providers without changing agent logic

---

## Project Structure

```
├── agent.py                    # Abstract Agent base class
├── response_service.py         # Abstract LLM response service
├── embedding_service.py        # Abstract embedding service
├── vector_db.py                # Abstract VectorDB base class
├── conversation_memory.py      # In-memory conversation history
├── llm_message.py              # Message dataclass (user/assistant/tool)
├── decorator.py                # @tool decorator + schema generation
├── singleton.py                # Thread-safe singleton decorator
├── env.py                      # .env loader (singleton)
│
├── openai_response_service.py  # OpenAI chat completions
├── openai_embedding_service.py # OpenAI embeddings
│
├── pg_vector_db.py             # PostgreSQL + pgvector backend
├── redis_vector_db.py          # Redis Stack vector search backend
│
├── tech_support_agent.py       # Example: RAG-powered support agent
├── retail_agent.py             # Example: retail customer service agent with tool use
├── audit_agent.py              # Example: prompt injection / policy auditor
│
└── docker-compose.yml          # Redis Stack + PostgreSQL services
```

---

## Provider Support

### LLM Response Providers

| Provider | Class | Status |
|---|---|---|
| OpenAI | `OpenAILlmResponseService` | ✅ Supported |

Default model: `gpt-4o`. Pass any OpenAI-compatible model string to the constructor.

```python
from src.llm.service.openai_response_service import OpenAILlmResponseService

llm = OpenAILlmResponseService(model="gpt-4o-mini")
```

### Embedding Providers

| Provider | Class | Models | Status |
|---|---|---|---|
| OpenAI | `OpenAILlmEmbeddingService` | `text-embedding-3-small`, `text-embedding-3-large` | ✅ Supported |

```python
from src.llm.service.openai_embedding_service import OpenAILlmEmbeddingService

embedder = OpenAILlmEmbeddingService(dimensions=300, model="text-embedding-3-small")
```

---

## Vector Database Support

| Database | Class | Backend | Status |
|---|---|---|---|
| PostgreSQL | `PGVectorDB` | `pgvector` extension | ✅ Supported |
| Redis | `RedisVectorDB` | Redis Stack (JSON + Search) | ✅ Supported |

### PostgreSQL (pgvector)

```python
from src.vector_db.pg_vector_db import PGVectorDB

db = PGVectorDB(
    embedding_service=embedder,
    host="localhost",
    port=5432,
    database="postgres",
    user="postgres",
    password="postgres",
    table_name="documents",
)
```

### Redis Stack

```python
from src.vector_db.redis_vector_db import RedisVectorDB

db = RedisVectorDB(
    embedding_service=embedder,
    host="localhost",
    port=6379,
    index_name="documents",
    vector_dim=300,
)
```

Start both databases locally with:

```bash
docker-compose up -d
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Add your OPENAI_API_KEY to .env
```

### 3. Run an example agent

```bash
# Tech support agent with RAG (requires Redis or Postgres running)
python -m src.agent.tech_support_agent

# Retail customer service agent
python -m src.agent.retail_agent

# Audit / prompt injection detection agent
python -m src.agent.audit_agent
```

---

## Building Your Own Agent

### 1. Define your tools

```python
from src.tools.decorator import tool
from typing import Dict, Any

@tool
def get_order_status(order_id: str) -> Dict[str, Any]:
    """
    Retrieve the current status of a customer order.

    :param order_id: The unique order identifier
    """
    # your logic here
    return {"order_id": order_id, "status": "shipped"}
```

The `@tool` decorator reads your type hints and `:param` docstring entries to automatically generate the OpenAI function-calling schema — no Pydantic required.

### 2. Create your agent

```python
from src.agent.agent import Agent
from src.llm.service.response_service import LlmResponseService
from src.tools.decorator import get_tool_definitions
from typing import Callable, List, Optional, Dict

class MyAgent(Agent):
    def __init__(self, llm_service: LlmResponseService):
        self.__llm_service = llm_service

    @property
    def agent_type(self) -> str:
        return "My Agent"

    def execute(self, task: str) -> str:
        response = self.__llm_service.response(
            role="You are a helpful assistant.",
            prompt=task,
            conversation_history=None,
        )
        return response["choices"][0]["message"]["content"]
```
