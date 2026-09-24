"""Pruebas automatizadas del sistema RAG (no usan API keys ni consumen tokens).

Se reemplazan el modelo de embeddings y el LLM por dobles de prueba deterministas para verificar
la lógica propia: chunking, persistencia, recuperación, validación Pydantic, respuestas fuera de
contexto y seguridad de credenciales.

Ejecutar:  python -m pytest -v
"""

import asyncio
import hashlib
import inspect
import json
import re
from pathlib import Path

import numpy as np
import pytest
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.fake_chat_models import FakeListChatModel

import config
import ingest
import rag

ROOT = Path(__file__).resolve().parents[1]


class BagOfWordsEmbeddings(Embeddings):
    """Embedding determinista (bolsa de palabras) para pruebas offline."""

    def _embed(self, text):
        v = np.zeros(512)
        for w in re.findall(r"\w+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % 512] += 1
        return (v / (np.linalg.norm(v) or 1)).tolist()

    def embed_documents(self, texts):
        return [self._embed(t) for t in texts]

    def embed_query(self, text):
        return self._embed(text)


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Base vectorial temporal con embeddings falsos y datos reales de /data."""
    monkeypatch.setattr(config, "PERSIST_DIR", str(tmp_path / "vectorstore"))
    monkeypatch.setattr(config, "DATA_DIR", str(ROOT / "data"))
    monkeypatch.setattr(config, "get_embeddings", lambda: BagOfWordsEmbeddings())
    monkeypatch.setattr(config, "embedding_model_id", lambda: "test/bag-of-words")
    monkeypatch.setattr(rag, "_chain", None)
    return tmp_path


def respuesta_llm(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


# ------------------------------------------------------------------ 1. chunking e ingesta
def test_chunks_respetan_tamano_y_tienen_metadatos(entorno):
    chunks = ingest.split_documents(ingest.load_documents())
    splitter_len = ingest.RecursiveCharacterTextSplitter.from_tiktoken_encoder(encoding_name="cl100k_base")._length_function
    assert len(chunks) > len(ingest.load_documents()), "los documentos deben fragmentarse"
    assert all(splitter_len(c.page_content) <= config.CHUNK_SIZE for c in chunks)
    assert all(c.metadata["source"].endswith(".md") for c in chunks)


def test_limpieza_de_texto():
    assert ingest.clean_text("Hola    mundo\n\n\n\nfin  ") == "Hola mundo\n\nfin"


def test_ingesta_no_reindexa_si_la_base_existe(entorno):
    store = ingest.ingest()
    total = store._collection.count()
    assert total > 0
    store2 = ingest.ingest()  # segunda ejecución
    assert store2._collection.count() == total, "no debe duplicar chunks"


def test_detecta_embeddings_no_coincidentes(entorno, monkeypatch):
    ingest.ingest()
    monkeypatch.setattr(config, "embedding_model_id", lambda: "otro/modelo")
    with pytest.raises(RuntimeError, match="--reset"):
        ingest.check_embedding_model(ingest.get_vectorstore())


# ------------------------------------------------------------------ 2. arquitectura asíncrona y recuperación
def test_get_rag_response_es_asincrona():
    assert inspect.iscoroutinefunction(rag.get_rag_response)


def test_retriever_devuelve_top_k_y_el_documento_correcto(entorno):
    docs = rag.get_retriever().invoke("diferencia entre get y query en ChromaDB upsert")
    assert len(docs) == config.TOP_K
    assert 3 <= config.TOP_K <= 5
    assert docs[0].metadata["source"] == "04_chromadb_y_rag.md"


# ------------------------------------------------------------------ 3. validación Pydantic y fuentes
def test_respuesta_en_contexto_incluye_fuentes_validas(entorno):
    llm = FakeListChatModel(responses=[respuesta_llm(
        respuesta="get busca por ID y query por similitud.",
        encontrado_en_contexto=True,
        fuentes=["04_chromadb_y_rag.md", "archivo_inventado.md"],
    )])
    chain = rag.build_rag_chain(rag.get_retriever(), llm)
    r = asyncio.run(rag.get_rag_response("¿Diferencia entre get y query en ChromaDB?", chain))
    assert isinstance(r, rag.RespuestaRAG)
    assert r.encontrado_en_contexto is True
    assert r.fuentes == ["04_chromadb_y_rag.md"], "se descartan fuentes que no fueron recuperadas"


def test_json_mal_formado_se_reintenta(entorno):
    llm = FakeListChatModel(responses=["{esto no es json", respuesta_llm(
        respuesta="El overlap preserva contexto.", encontrado_en_contexto=True, fuentes=["03_embeddings_y_chunking.md"])])
    chain = rag.build_rag_chain(rag.get_retriever(), llm)
    r = asyncio.run(rag.get_rag_response("¿Para qué sirve el chunk_overlap?", chain))
    assert r.respuesta == "El overlap preserva contexto."


# ------------------------------------------------------------------ 4. robustez: preguntas fuera de contexto
def test_pregunta_fuera_de_contexto_responde_no_lo_se(entorno):
    # Aunque el modelo "sepa" la respuesta, si no está en el contexto el contrato fuerza "No lo sé"
    llm = FakeListChatModel(responses=[respuesta_llm(
        respuesta="Canberra", encontrado_en_contexto=False, fuentes=["01_lcel_y_runnables.md"])])
    chain = rag.build_rag_chain(rag.get_retriever(), llm)
    r = asyncio.run(rag.get_rag_response("¿Cuál es la capital de Australia?", chain))
    assert r.respuesta == rag.NO_LO_SE
    assert r.fuentes == []


def test_prompt_exige_no_lo_se_y_solo_contexto():
    system = rag.prompt.format_messages(contexto="x", pregunta="y")[0].content
    assert "No lo sé" in system and "ÚNICAMENTE" in system


# ------------------------------------------------------------------ 5. seguridad de credenciales
def test_env_esta_en_gitignore():
    reglas = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
    assert ".env" in reglas and "vectorstore/" in reglas


def test_no_hay_api_keys_en_el_codigo():
    patron = re.compile(r"(sk-[A-Za-z0-9_-]{20,}|AIza[0-9A-Za-z_-]{30,})")
    archivos = [p for p in ROOT.rglob("*") if p.suffix in {".py", ".md", ".txt", ".example"}
                and ".venv" not in p.parts and "vectorstore" not in p.parts]
    filtradas = [str(p.relative_to(ROOT)) for p in archivos if patron.search(p.read_text(encoding="utf-8", errors="ignore"))]
    assert not filtradas, f"posibles API keys en: {filtradas}"


def test_env_example_solo_tiene_placeholders():
    contenido = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "tu_api_key" in contenido and "sk-" not in contenido and "AIza" not in contenido
