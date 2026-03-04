import psycopg2
import psycopg2.extras
from typing import override, Optional, Any
from src.llm.service.embedding_service import LlmEmbeddingService
from src.vector_db.vector_db import VectorDB


class PGVectorDB(VectorDB):
    def __init__(
            self,
            embedding_service: LlmEmbeddingService,
            host: str = "localhost",
            port: int = 5432,
            database: str = "postgres",
            user: str = "postgres",
            password: str = "",
            table_name: str = "documents",
    ):
        super().__init__(embedding_service)
        self.__table_name = table_name
        self.__conn = psycopg2.connect(
            host=host,
            port=port,
            dbname=database,
            user=user,
            password=password
        )
        self.__ensure_table()

    def __ensure_table(self) -> None:
        with self.__conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute(f"""
                CREATE TABLE IF NOT EXISTS {self.__table_name} (
                    id      TEXT PRIMARY KEY,
                    text    TEXT NOT NULL,
                    embedding VECTOR
                )
            """)
        self.__conn.commit()

    @override
    def add_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        embeddings = self.__embedding_service.embed_batch(documents)
        with self.__conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, f"""
                INSERT INTO {self.__table_name} (id, text, embedding)
                VALUES (%s, %s, %s)
                ON CONFLICT (id) DO NOTHING
            """, [(doc_id, text, str(embedding)) for doc_id, text, embedding in zip(document_ids, documents, embeddings)])
        self.__conn.commit()

    @override
    def update_documents(
            self,
            document_ids: list[str],
            documents: list[str],
    ) -> None:
        embeddings = self.__embedding_service.embed_batch(documents)
        with self.__conn.cursor() as cur:
            psycopg2.extras.execute_batch(cur, f"""
                UPDATE {self.__table_name}
                SET text = %s, embedding = %s
                WHERE id = %s
            """, [(text, str(embedding), doc_id) for doc_id, text, embedding in zip(document_ids, documents, embeddings)])
        self.__conn.commit()

    @override
    def delete_documents(
            self,
            document_ids: list[str],
    ) -> None:
        with self.__conn.cursor() as cur:
            cur.execute(f"""
                DELETE FROM {self.__table_name}
                WHERE id = ANY(%s)
            """, (document_ids,))
        self.__conn.commit()

    @override
    def similarity_search(
            self,
            query: str,
            top_k: Optional[int] = 5,
    ) -> list[Any]:
        query_vector = self.__embedding_service.embed(query)
        with self.__conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(f"""
                SELECT id, text, 1 - (embedding <=> %s::vector) AS score
                FROM {self.__table_name}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (str(query_vector), str(query_vector), top_k))
            return cur.fetchall()

    def close(self) -> None:
        self.__conn.close()
