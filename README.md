# Pre-entrega 3 · Sistema de recuperación semántica local (RAG)

AI Engineering (Coderhouse) · Módulo 3 — Embeddings, chunking y persistencia con ChromaDB.

Flujo RAG *end-to-end*: indexa apuntes del curso en una base vectorial local (ChromaDB) y responde preguntas
usando **exclusivamente** esa información. Si la respuesta no está en los documentos, responde **"No lo sé"**.

## Estructura

| Archivo | Qué contiene |
|---|---|
| `config.py` | Configuración compartida: rutas, `chunk_size`, `top_k` y **un único** modelo de embeddings y LLM |
| `ingest.py` | **Ingesta**: carga `.md`/`.txt`, limpia, fragmenta con `RecursiveCharacterTextSplitter` y persiste en ChromaDB |
| `rag.py` | **Retriever + generación grounded**: cadena LCEL asíncrona con `PydanticOutputParser` |
| `data/` | Dataset de ejemplo: 4 apuntes del curso (LCEL, validación, embeddings/chunking, ChromaDB/RAG) |

## Arquitectura

```
INGESTA (ingest.py)
data/*.md ──► limpieza ──► RecursiveCharacterTextSplitter ──► embeddings ──► ChromaDB (./vectorstore)
                           (200 tokens, overlap 40, tiktoken)   (gemini-embedding-001)   IDs = hash del contenido

CONSULTA (rag.py)
pregunta ──► retriever (top_k=4, coseno) ──► format_docs ──┐
    └──────────────────────────────────────────────────────┴──► prompt "filtro de veracidad" ──► LLM ──► PydanticOutputParser ──► RespuestaRAG
```

## Componentes pedidos por la consigna

**1. Módulo de ingesta (`ingest.py`)**
- Limpia el texto (espacios y saltos de línea repetidos) antes de fragmentar.
- `RecursiveCharacterTextSplitter.from_tiktoken_encoder`: chunks de 200 **tokens** con 40 de overlap.
- Persiste en `./vectorstore` con `langchain_chroma.Chroma` (métrica coseno).
- **IDs deterministas** (hash SHA-256 de fuente + contenido): antes de indexar, verifica qué chunks ya existen y
  solo embebe los nuevos. Si la base ya está al día, **no reindexa** (ahorra tiempo y costo).
- `python ingest.py --reset` borra la colección y reindexa todo.

**2. Capa de recuperación (`rag.py`)**
- `vectorstore.as_retriever(search_kwargs={"k": 4})`: convierte la pregunta en embedding y trae los 4 fragmentos
  más similares (dentro del rango recomendado de 3 a 5, para evitar el efecto *Lost in the Middle*).
- **Mismo modelo de embeddings para indexar y consultar**: está definido una sola vez en `config.py`, se guarda
  como metadato de la colección y `check_embedding_model()` corta la ejecución si no coincide.

**3. Generación grounded (`rag.py`)**
- Cadena LCEL: `RunnableParallel(contexto=retriever | format_docs, pregunta) | prompt | llm | PydanticOutputParser`.
- Prompt de sistema como "filtro de veracidad": responde solo con el CONTEXTO y, si no está, dice **"No lo sé"**.
- Salida validada con Pydantic (`RespuestaRAG`): `respuesta`, `encontrado_en_contexto` y `fuentes`.
  Un `model_validator` garantiza coherencia: si no se encontró, la respuesta es "No lo sé" y no hay fuentes.
- Ejecución **asíncrona** (`ainvoke`); las preguntas de la demo corren en paralelo con `asyncio.gather`.
- `with_retry` reintenta si el modelo devuelve un JSON mal formado.

## Cómo ejecutarlo

```bash
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env            # Mac/Linux: cp .env.example .env  → completá tu key

python ingest.py                  # 1) indexa los apuntes en ./vectorstore
python ingest.py                  # 2) volver a correrlo NO reindexa (la base ya existe)
python rag.py                     # 3) demo con 4 preguntas (una fuera de contexto)
python rag.py "¿Qué hace upsert en ChromaDB?"   # pregunta propia
```

Con `GOOGLE_API_KEY` usa Gemini (`gemini-embedding-001` + `gemini-2.5-flash`); con `OPENAI_API_KEY`,
OpenAI (`text-embedding-3-small` + `gpt-4o-mini`).

## Ejemplo de salida

_(se completa con la corrida real)_
