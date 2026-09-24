# Pre-entrega 3 · Sistema de recuperación semántica local (RAG)

AI Engineering (Coderhouse) · Módulo 3 — Embeddings, chunking y persistencia con ChromaDB.

Flujo RAG *end-to-end*: indexa apuntes del curso en una base vectorial local (ChromaDB) y responde preguntas
usando **exclusivamente** esa información. Si la respuesta no está en los documentos, responde **"No lo sé"**.

## Cumplimiento de los criterios de evaluación

| Criterio (peso) | Dónde se cumple |
|---|---|
| **Chunking e ingesta (25%)** | `ingest.py`: procesa `/data`, limpia el texto y usa `RecursiveCharacterTextSplitter` por tokens (200, overlap 40) con separadores párrafo → línea → oración para preservar el contexto semántico |
| **Arquitectura asíncrona y recuperación (35%)** | `rag.py`: `async def get_rag_response(pregunta)` usa `ainvoke` sobre la cadena LCEL (retriever + LLM); la demo ejecuta varias preguntas en paralelo con `asyncio.gather` |
| **Validación con Pydantic (20%)** | `RespuestaRAG` (`respuesta`, `encontrado_en_contexto`, `fuentes`) vía `PydanticOutputParser`; las fuentes se verifican contra los documentos realmente recuperados |
| **Robustez y evidencia de pruebas (20%)** | Preguntas fuera de contexto → "No lo sé" (probado en la corrida real y en tests); `.env` fuera del repo; 13 tests con `pytest` |

## Estructura

| Archivo | Qué contiene |
|---|---|
| `config.py` | Configuración compartida: rutas, `chunk_size`, `top_k` y **un único** modelo de embeddings y LLM |
| `ingest.py` | **Ingesta**: carga `.md`/`.txt`, limpia, fragmenta con `RecursiveCharacterTextSplitter` y persiste en ChromaDB |
| `rag.py` | **Retriever + generación grounded**: `get_rag_response()` asíncrona, cadena LCEL con `PydanticOutputParser` |
| `tests/test_rag.py` | **13 pruebas automatizadas** (pytest), offline y sin API key |
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
- Cadena LCEL: `assign(docs=retriever) | assign(respuesta=format_docs | prompt | llm | PydanticOutputParser) | verificar_fuentes`.
- Prompt de sistema como "filtro de veracidad": responde solo con el CONTEXTO y, si no está, dice **"No lo sé"**.
- Salida validada con Pydantic (`RespuestaRAG`): `respuesta`, `encontrado_en_contexto` y `fuentes`.
  Un `model_validator` garantiza coherencia: si no se encontró, la respuesta es "No lo sé" y no hay fuentes.
- Punto de entrada **`async def get_rag_response(pregunta) -> RespuestaRAG`**: usa `ainvoke`; las preguntas de la demo corren en paralelo con `asyncio.gather`.
- `verificar_fuentes()` descarta fuentes que el modelo cite pero que no hayan sido recuperadas (evita fuentes inventadas).
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
python -m pytest -v               # 4) pruebas automatizadas (no usan la API)
```

Uso desde otro módulo:

```python
import asyncio
from rag import get_rag_response

r = asyncio.run(get_rag_response("¿Qué es el chunk_overlap?"))
print(r.respuesta, r.fuentes)
```

Con `GOOGLE_API_KEY` usa Gemini (`gemini-embedding-001` + `gemini-2.5-flash`); con `OPENAI_API_KEY`,
OpenAI (`text-embedding-3-small` + `gpt-4o-mini`).

## Ejemplo de salida (corrida real con Gemini)

**Ingesta** — primera y segunda ejecución:

```
$ python ingest.py
INFO | Documentos cargados: 4
INFO | Chunks generados: 10 (chunk_size=200 tokens, overlap=40)
INFO | HTTP Request: POST .../gemini-embedding-001:batchEmbedContents "HTTP/1.1 200 OK"
INFO | Chunks nuevos indexados: 10 | Total en la colección: 10

$ python ingest.py
INFO | Documentos cargados: 4
INFO | Chunks generados: 10 (chunk_size=200 tokens, overlap=40)
INFO | La base ya está al día (10 chunks). No se reindexa nada.
```

La segunda ejecución no vuelve a llamar a la API de embeddings: los IDs deterministas detectan que la base ya existe.

**Consulta** — `python rag.py` (4 preguntas en paralelo, la última fuera de contexto):

```
❓ ¿Qué diferencia hay entre get y query en ChromaDB?
💬 En ChromaDB, `get` busca registros por IDs exactos o filtros de metadatos, funcionando de manera similar a un
   `SELECT` tradicional. Por otro lado, `query` busca por similitud semántica, lo cual es fundamental para RAG.
   encontrado_en_contexto=True | fuentes=['04_chromadb_y_rag.md']

❓ ¿Para qué sirve el chunk_overlap?
💬 El `chunk_overlap` repite parte del final de un fragmento (chunk) al inicio del siguiente para evitar perder
   contexto en los cortes.
   encontrado_en_contexto=True | fuentes=['03_embeddings_y_chunking.md']

❓ ¿Cuántos fragmentos conviene recuperar en un RAG y por qué?
💬 Se recomienda usar un `top_k` de entre 3 y 5 fragmentos en un sistema RAG. Esto se debe a que pasar demasiados
   fragmentos puede provocar errores de límite de tokens o el efecto "Lost in the Middle", donde el modelo ignora
   la información ubicada en el medio del contexto.
   encontrado_en_contexto=True | fuentes=['04_chromadb_y_rag.md']

❓ ¿Cuál es la capital de Australia?
💬 No lo sé
   encontrado_en_contexto=False | fuentes=[]
```

- Las 3 preguntas del temario se responden con el contenido de los apuntes y citan el archivo correcto.
- La pregunta fuera de dominio responde **"No lo sé"** aunque el modelo conoce la respuesta: el prompt
  "filtro de veracidad" y el validador de `RespuestaRAG` impiden usar conocimiento externo.

## Pruebas automatizadas

`python -m pytest -v` ejecuta 13 pruebas **sin API key ni consumo de tokens**: el modelo de embeddings y el LLM se
reemplazan por dobles de prueba deterministas, y se usan los apuntes reales de `/data` en una base temporal.

| Área | Qué se verifica |
|---|---|
| Chunking e ingesta | Los chunks respetan `chunk_size` en tokens y tienen `source`; la limpieza normaliza el texto; la segunda ingesta no duplica; se detecta un modelo de embeddings distinto |
| Asincronía y recuperación | `get_rag_response` es una corrutina; el retriever devuelve `top_k` (3–5) fragmentos y el primero es el apunte correcto |
| Pydantic y fuentes | La salida es `RespuestaRAG`; se descartan fuentes inventadas; un JSON mal formado se reintenta |
| Fuera de contexto | Aunque el modelo "sepa" la respuesta, si no está en el contexto se devuelve "No lo sé" sin fuentes; el prompt lo exige |
| Credenciales | `.env` está en `.gitignore`; no hay API keys en el código; `.env.example` solo tiene placeholders |

## Seguridad de credenciales

- Las keys se leen de `.env` con `python-dotenv`; **`.env` está en `.gitignore`** y nunca se sube.
- `.env.example` documenta las variables necesarias con valores de ejemplo.
- `vectorstore/` también se ignora: se regenera localmente con `python ingest.py`.
- Un test (`test_no_hay_api_keys_en_el_codigo`) falla si alguien sube por error una key de OpenAI o Google.
