"""Cadena RAG asíncrona (LCEL): retriever de ChromaDB + generación grounded + PydanticOutputParser.

Uso:
    python rag.py                         # demo con preguntas de ejemplo (incluye una fuera de contexto)
    python rag.py "¿Qué hace upsert?"     # pregunta propia
"""

import asyncio
import logging
import sys
from operator import itemgetter
from typing import List

from langchain_core.documents import Document
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda, RunnableParallel
from pydantic import BaseModel, Field, model_validator

import config
from ingest import check_embedding_model, get_vectorstore, ingest

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger("rag")

NO_LO_SE = "No lo sé"


# ---------------------------------------------------------------- contrato de salida
class RespuestaRAG(BaseModel):
    """Respuesta del asistente, basada exclusivamente en el contexto recuperado."""

    respuesta: str = Field(description=f"Respuesta en español basada SOLO en el contexto, o exactamente '{NO_LO_SE}'.")
    encontrado_en_contexto: bool = Field(description="true si la respuesta está en el contexto; false si no.")
    fuentes: List[str] = Field(default_factory=list,
                               description="Nombres de archivo (source) de los fragmentos usados. Vacía si no se encontró.")

    @model_validator(mode="after")
    def coherencia(self):
        # Si no está en el contexto, la respuesta es "No lo sé" y no se citan fuentes
        if not self.encontrado_en_contexto:
            self.respuesta = NO_LO_SE
            self.fuentes = []
        return self


parser = PydanticOutputParser(pydantic_object=RespuestaRAG)

# ---------------------------------------------------------------- prompt "filtro de veracidad"
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Sos un asistente técnico del curso AI Engineering. Respondé ÚNICAMENTE con la información del CONTEXTO.\n"
        "Reglas:\n"
        "- No uses conocimiento propio ni inventes datos.\n"
        f"- Si la respuesta no está en el CONTEXTO, respondé exactamente \"{NO_LO_SE}\" y marcá "
        "encontrado_en_contexto como false.\n"
        "- En fuentes, listá el nombre de archivo de cada fragmento que usaste.\n\n"
        "{format_instructions}",
    ),
    ("human", "CONTEXTO:\n{contexto}\n\nPREGUNTA: {pregunta}"),
]).partial(format_instructions=parser.get_format_instructions())


def format_docs(docs: List[Document]) -> str:
    """Transforma los documentos recuperados en un bloque de texto con su fuente."""
    if not docs:
        return "(sin fragmentos relevantes)"
    return "\n\n".join(f"[fuente: {d.metadata.get('source', '?')}]\n{d.page_content}" for d in docs)


# ---------------------------------------------------------------- cadena LCEL
def build_rag_chain(retriever: Runnable, llm=None) -> Runnable:
    llm = llm or config.get_llm()
    chain = (
        RunnableParallel(
            contexto=itemgetter("pregunta") | retriever | RunnableLambda(format_docs),
            pregunta=itemgetter("pregunta"),
        )
        | prompt
        | llm
        | parser
    )
    # Si el modelo devuelve un JSON mal formado, reintenta una vez más
    return chain.with_retry(retry_if_exception_type=(OutputParserException,), stop_after_attempt=2)


def get_retriever():
    store = get_vectorstore()
    if store._collection.count() == 0:
        logger.info("La base vectorial está vacía: ejecutando la ingesta primero")
        store = ingest()
    check_embedding_model(store)  # mismo modelo de embeddings para indexar y consultar
    return store.as_retriever(search_kwargs={"k": config.TOP_K})


async def preguntar(chain: Runnable, pregunta: str) -> RespuestaRAG:
    return await chain.ainvoke({"pregunta": pregunta})


async def main(preguntas: List[str]):
    chain = build_rag_chain(get_retriever())
    # Las preguntas son independientes: se ejecutan en paralelo
    resultados = await asyncio.gather(*(preguntar(chain, p) for p in preguntas), return_exceptions=True)
    for pregunta, r in zip(preguntas, resultados):
        print(f"\n❓ {pregunta}")
        if isinstance(r, Exception):
            print(f"   ⚠ Error ({type(r).__name__}): {r}")
            continue
        print(f"💬 {r.respuesta}")
        print(f"   encontrado_en_contexto={r.encontrado_en_contexto} | fuentes={r.fuentes}")


PREGUNTAS_DEMO = [
    "¿Qué diferencia hay entre get y query en ChromaDB?",
    "¿Para qué sirve el chunk_overlap?",
    "¿Cuántos fragmentos conviene recuperar en un RAG y por qué?",
    "¿Cuál es la capital de Australia?",  # fuera del contexto -> debe responder "No lo sé"
]

if __name__ == "__main__":
    preguntas = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else PREGUNTAS_DEMO
    asyncio.run(main(preguntas))
