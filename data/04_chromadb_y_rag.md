# Apunte 4 · ChromaDB y RAG

## ChromaDB
ChromaDB es una base de datos vectorial de código abierto. Con `PersistentClient(path=...)` guarda los datos en
una carpeta local, de modo que el conocimiento sobrevive al reinicio del script.

Operaciones CRUD:
- `upsert`: crea el registro si el ID no existe o lo actualiza si ya existe. Es la mejor práctica.
- `get`: busca por IDs exactos o filtros de metadatos, como un SELECT tradicional.
- `query`: busca por similitud semántica; es el corazón de RAG.
- `delete`: elimina vectores que ya no son relevantes.

Los metadatos (por ejemplo `source` o `fecha`) permiten filtrar resultados y citar las fuentes. Se recomiendan IDs
deterministas, como un hash del contenido, para poder actualizar sin duplicar.

## RAG (Retrieval-Augmented Generation)
Un sistema RAG convierte la pregunta del usuario en un embedding, recupera los fragmentos más similares de la base
vectorial y se los pasa al LLM como contexto para generar la respuesta.

Recomendaciones:
- Usar un `top_k` de entre 3 y 5 fragmentos. Pasar demasiados provoca errores de límite de tokens o el efecto
  "Lost in the Middle", en el que el modelo ignora la información ubicada en el medio del contexto.
- Usar el mismo modelo de embeddings para indexar y para consultar.
- Instruir al modelo para que responda "No lo sé" cuando la respuesta no está en el contexto (generación grounded).
- Verificar si la base ya existe antes de reindexar, para ahorrar tiempo y costo.
