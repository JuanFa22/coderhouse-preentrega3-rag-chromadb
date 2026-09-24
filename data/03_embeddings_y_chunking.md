# Apunte 3 · Embeddings, similitud y chunking

## Embeddings
Un embedding es un vector denso de números que representa el significado de un texto. Textos con significado
parecido quedan cerca en el espacio vectorial aunque no compartan palabras. Los modelos modernos son contextuales:
la palabra "banco" genera vectores distintos según hable de un asiento o de una entidad financiera.

## Métricas de similitud
- Similitud coseno: mide el ángulo entre dos vectores e ignora su longitud. Es la métrica estándar en RAG.
- Distancia euclídea (L2): mide la distancia en línea recta y es sensible a la magnitud.
- Producto punto: combina magnitud y ángulo; se usa en sistemas de recomendación.

Nunca se deben comparar embeddings generados por modelos distintos: viven en espacios vectoriales diferentes.
Los embeddings también pueden confundir negaciones, como "hay stock" y "no hay stock".

## Chunking
Los documentos largos se dividen en fragmentos (chunks) porque un único vector de un documento extenso diluye
su significado. `RecursiveCharacterTextSplitter` intenta cortar primero por párrafos, luego por líneas, espacios y
finalmente caracteres. El `chunk_overlap` repite parte del final de un chunk al inicio del siguiente para no
perder contexto en los cortes. Conviene medir los chunks en tokens con un tokenizador real como tiktoken.

Antes de fragmentar hay que limpiar el texto: encabezados repetidos, saltos de línea múltiples y caracteres basura.
