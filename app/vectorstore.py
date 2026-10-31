from functools import lru_cache
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from app import config


@lru_cache
def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    return Chroma(
        collection_name="docs",
        embedding_function=embeddings,
        persist_directory=config.CHROMA_DIR,
    )


def delete_source(source: str) -> None:
    """Remove all chunks of a source so re-ingesting never creates duplicates."""
    get_vectorstore()._collection.delete(where={"source": source})
