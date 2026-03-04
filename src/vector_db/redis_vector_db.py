import json
import struct
import redis
import numpy as np
from typing import override, Optional, Any
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

    import time

    def __ensure_index(self) -> None:
        try:
            self.__client.ft(self.__index_name).dropindex()
        except Exception:
            pass

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
        self.__wait_for_indexing()

    def __wait_for_indexing(self, timeout: int = 30) -> None:
        import time
        start = time.time()
        while time.time() - start < timeout:
            info = self.__client.ft(self.__index_name).info()
            print(f"indexing={info['indexing']}, num_docs={info['num_docs']}, percent={info.get('percent_indexed')}")
            if int(info["indexing"]) == 0 and int(info["num_docs"]) > 0:
                return
            time.sleep(0.1)
        raise TimeoutError("Redis Search indexing did not complete in time")

    @override
    def add_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        embeddings = self._embedding_service.embed_batch(documents)
        pipe = self.__client.pipeline(transaction=False)
        for doc_id, text, embedding in zip(document_ids, documents, embeddings):
            key = f"{self.__prefix}{doc_id}"
            pipe.json().set(key, "$", {
                "text": text,
                "embedding": embedding
            })
        pipe.execute()
        self.__ensure_index()

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
        query_vector = self._embedding_service.embed(query)
        query_vector = np.array(query_vector, dtype=np.float32).tobytes()

        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("text", "score")
            .dialect(2)
        )

        results = self.__client.ft(self.__index_name).search(
            q, query_params={"vec": query_vector}
        )

        return [
            {"id": doc.id.replace(self.__prefix, ""), "text": doc.text, "score": doc.score}
            for doc in results.docs
        ]
