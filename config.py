"""Configuración compartida por la ingesta y la consulta.

Tener el modelo de embeddings definido en UN solo lugar evita el error #1 de RAG:
indexar con un modelo y consultar con otro.
"""

import os

from dotenv import load_dotenv

load_dotenv()

DATA_DIR = "data"
PERSIST_DIR = "vectorstore"
COLLECTION_NAME = "apuntes_ai_engineering"

CHUNK_SIZE = 200      # tokens por chunk
CHUNK_OVERLAP = 40    # tokens compartidos entre chunks consecutivos
TOP_K = 4             # fragmentos recuperados (recomendado: 3 a 5)


def get_provider() -> str:
    if os.getenv("GOOGLE_API_KEY"):
        return "google"
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    raise RuntimeError("Falta GOOGLE_API_KEY u OPENAI_API_KEY en el archivo .env")


def get_embeddings():
    """Modelo de embeddings: el MISMO para indexar y para consultar."""
    if get_provider() == "google":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        return GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")


def embedding_model_id() -> str:
    """Identificador del modelo de embeddings, guardado como metadato de la colección."""
    return "google/gemini-embedding-001" if get_provider() == "google" else "openai/text-embedding-3-small"


def get_llm():
    if get_provider() == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0)
