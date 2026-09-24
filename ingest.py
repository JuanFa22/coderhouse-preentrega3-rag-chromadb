"""Módulo de ingesta: carga documentos .txt/.md, los limpia, los fragmenta y los persiste en ChromaDB.

Uso:
    python ingest.py            # indexa solo lo nuevo (si la base ya existe, no reindexa lo que ya está)
    python ingest.py --reset    # borra la colección y reindexa todo
"""

import hashlib
import logging
import re
import sys
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("ingest")


# ---------------------------------------------------------------- 1. carga y limpieza
def clean_text(text: str) -> str:
    """Normaliza espacios y saltos de línea sin romper los párrafos."""
    text = text.replace("\r\n", "\n").replace("\t", " ")
    text = re.sub(r"[  ]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(data_dir: str = config.DATA_DIR) -> List[Document]:
    docs = []
    for path in sorted(Path(data_dir).glob("*")):
        if path.suffix.lower() in {".txt", ".md"}:
            docs.append(Document(page_content=clean_text(path.read_text(encoding="utf-8")),
                                 metadata={"source": path.name}))
    if not docs:
        raise FileNotFoundError(f"No hay archivos .txt o .md en '{data_dir}'")
    logger.info("Documentos cargados: %d", len(docs))
    return docs


# ---------------------------------------------------------------- 2. chunking
def split_documents(docs: List[Document]) -> List[Document]:
    """RecursiveCharacterTextSplitter midiendo en tokens (tiktoken) con overlap."""
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        encoding_name="cl100k_base",
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk"] = i
    logger.info("Chunks generados: %d (chunk_size=%d tokens, overlap=%d)",
                len(chunks), config.CHUNK_SIZE, config.CHUNK_OVERLAP)
    return chunks


def chunk_id(chunk: Document) -> str:
    """ID determinista: hash del origen + contenido. Permite upsert sin duplicar."""
    raw = f"{chunk.metadata['source']}::{chunk.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------- 3. persistencia
def get_vectorstore() -> Chroma:
    """Abre (o crea) la colección persistente, registrando el modelo de embeddings usado."""
    return Chroma(
        collection_name=config.COLLECTION_NAME,
        embedding_function=config.get_embeddings(),
        persist_directory=config.PERSIST_DIR,
        collection_metadata={"hnsw:space": "cosine", "embedding_model": config.embedding_model_id()},
    )


def check_embedding_model(store: Chroma) -> None:
    """Evita consultar con un modelo de embeddings distinto al que se usó para indexar."""
    indexed_with = (store._collection.metadata or {}).get("embedding_model")
    current = config.embedding_model_id()
    if store._collection.count() and indexed_with and indexed_with != current:
        raise RuntimeError(
            f"La base fue indexada con '{indexed_with}' pero ahora se usa '{current}'. "
            "Ejecutá: python ingest.py --reset"
        )


def ingest(reset: bool = False) -> Chroma:
    store = get_vectorstore()
    if reset and store._collection.count():
        logger.info("--reset: borrando la colección existente")
        store.delete_collection()
        store = get_vectorstore()

    check_embedding_model(store)

    chunks = split_documents(load_documents())
    ids = [chunk_id(c) for c in chunks]

    # Verificar qué ya existe para no volver a embeber (ahorra tiempo y costo)
    existing = set(store.get(ids=ids, include=[])["ids"]) if store._collection.count() else set()
    nuevos = [(i, c) for i, c in zip(ids, chunks) if i not in existing]

    if not nuevos:
        logger.info("La base ya está al día (%d chunks). No se reindexa nada.", store._collection.count())
        return store

    store.add_documents(documents=[c for _, c in nuevos], ids=[i for i, _ in nuevos])
    logger.info("Chunks nuevos indexados: %d | Total en la colección: %d",
                len(nuevos), store._collection.count())
    return store


if __name__ == "__main__":
    ingest(reset="--reset" in sys.argv)
