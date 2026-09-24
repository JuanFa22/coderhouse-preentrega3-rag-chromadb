# Apunte 2 · Validación estructurada y resiliencia

## Contratos de datos con Pydantic
Los LLM producen texto libre, lo que es frágil para integrarlo con otros sistemas. Un esquema Pydantic funciona
como contrato de datos y valida tres niveles:
- Sintáctico: la respuesta es un JSON íntegro y parseable.
- Estructural: están presentes todos los campos requeridos con el tipo correcto.
- Semántico: los valores cumplen reglas de negocio, por ejemplo un puntaje entre 0 y 1 con `Field(ge=0, le=1)`.

Con `@field_validator` se pueden agregar reglas propias, como exigir que una lista no esté vacía o normalizar strings.

## Salida estructurada
El método `with_structured_output(Esquema)` usa tool calling o JSON mode del proveedor para que el modelo
devuelva directamente una instancia del esquema. Alternativamente, `PydanticOutputParser` genera instrucciones
de formato con `get_format_instructions()` y parsea la respuesta de texto.

## Resiliencia
- `with_retry(stop_after_attempt=3, wait_exponential_jitter=True)` reintenta con backoff exponencial ante errores
  transitorios como caídas de red o el error 429 de rate limit.
- `with_fallbacks([otro_modelo])` deriva la carga a un modelo alternativo si el principal falla de forma persistente.
- No conviene reintentar errores permanentes (por ejemplo un prompt mal formado): solo agregan costo y latencia.
- Hay que revisar el `finish_reason`: si vale `length` o `MAX_TOKENS`, la respuesta se cortó y el objeto está incompleto.
