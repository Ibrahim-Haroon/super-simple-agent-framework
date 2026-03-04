import json
import struct
from typing import override, Optional, Any
import redis
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
from src.llm.service.embedding_service import LlmEmbeddingService
from src.vector_db.vector_db import VectorDB


class RedisVectorDB(VectorDB):
    def __init__(
            self,
            embedding_service: LlmEmbeddingService,
            host: str = "localhost",
            port: int = 6379,
            index_name: str = "documents",
            vector_dim: int = 1536,
    ):
        super().__init__(embedding_service)
        self.__client = redis.Redis(host=host, port=port, decode_responses=False)
        self.__index_name = index_name
        self.__vector_dim = vector_dim
        self.__prefix = f"{index_name}:"
        self.__ensure_index()

    def __ensure_index(self) -> None:
        try:
            self.__client.ft(self.__index_name).info()
        except Exception:
            schema = (
                TextField("$.text", as_name="text"),
                VectorField(
                    "$.embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": self.__vector_dim,
                        "DISTANCE_METRIC": "COSINE",
                    },
                    as_name="embedding"
                ),
            )
            self.__client.ft(self.__index_name).create_index(
                schema,
                definition=IndexDefinition(
                    prefix=[self.__prefix],
                    index_type=IndexType.JSON
                )
            )

    @staticmethod
    def __to_float32_bytes(vector: list[float]) -> bytes:
        return struct.pack(f"{len(vector)}f", *vector)

    @override
    def add_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        embeddings = self.__embedding_service.embed_batch(documents)
        pipe = self.__client.pipeline()
        for doc_id, text, embedding in zip(document_ids, documents, embeddings):
            key = f"{self.__prefix}{doc_id}"
            self.__client.json().set(key, "$", {
                "text": text,
                "embedding": embedding
            })
        pipe.execute()

    @override
    def update_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        self.add_documents(document_ids, documents)

    @override
    def delete_documents(
            self,
            document_ids: list[str],
    ) -> None:
        pipe = self.__client.pipeline()
        for doc_id in document_ids:
            pipe.delete(f"{self.__prefix}{doc_id}")
        pipe.execute()

    @override
    def similarity_search(
            self,
            query: str,
            top_k: Optional[int] = 5,
    ) -> list[Any]:
        query_vector = self.__embedding_service.embed(query)
        query_bytes = self.__to_float32_bytes(query_vector)

        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("text", "score")
            .dialect(2)
        )

        results = self.__client.ft(self.__index_name).search(
            q, query_params={"vec": query_bytes}
        )

        return [
            {"id": doc.id.replace(self.__prefix, ""), "text": doc.text, "score": doc.score}
            for doc in results.docs
        ]
