# Apunte 1 · LCEL y el protocolo Runnable

## Qué es LCEL
LangChain Expression Language (LCEL) es una forma declarativa de componer cadenas de IA. Cada componente
(prompt, modelo, parser, retriever) se conecta con el operador pipe `|`: la salida del bloque de la izquierda
es la entrada del bloque de la derecha. La cadena típica es `prompt | modelo | parser`.

## El protocolo Runnable
Todos los componentes de LangChain implementan la interfaz Runnable, que expone los mismos métodos:
- `invoke` para una entrada única y `ainvoke` para su versión asíncrona.
- `batch` y `abatch` para procesar listas de entradas de forma optimizada.
- `stream` y `astream` para recibir la respuesta en fragmentos a medida que se genera.

Gracias a esta interfaz común, una cadena LCEL obtiene soporte async, streaming, ejecución por lotes y
observabilidad con LangSmith sin código adicional.

## Paralelismo
`RunnableParallel` ejecuta varias sub-cadenas independientes al mismo tiempo sobre la misma entrada y devuelve
un diccionario con una clave por rama. El tiempo total es el de la rama más lenta, no la suma de todas.

## Buenas prácticas
- En aplicaciones web como FastAPI conviene usar los métodos asíncronos (`ainvoke`, `abatch`) para no bloquear
  el event loop.
- Las variables del prompt (por ejemplo `{pregunta}`) deben coincidir exactamente con las claves del diccionario
  de entrada.
- Sin un output parser, la cadena devuelve un objeto `AIMessage` en lugar de texto plano.
