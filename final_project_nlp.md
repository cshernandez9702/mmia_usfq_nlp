# Construcción de un Chatbot con Generación Aumentada por Recuperación (RAG) usando Llama 3 para Consultas Locales en PDFs

## Introducción

En la actualidad, las organizaciones necesitan extraer información rápidamente de grandes volúmenes de documentos, incluyendo PDFs confidenciales. Los métodos tradicionales pueden ser ineficientes y propensos a errores, como las alucinaciones en modelos de lenguaje. Para mejorar la productividad y reducir estas alucinaciones, se desarrolla un chatbot que utiliza la **Generación Aumentada por Recuperación (RAG)** y **Llama 3** para permitir consultas eficientes y seguras de documentos PDF locales. Este proyecto demuestra cómo las técnicas avanzadas de Procesamiento de Lenguaje Natural (PLN) pueden crear una herramienta poderosa para extraer información de datos no estructurados, garantizando la confidencialidad y aumentando la precisión de las respuestas.


<p align="center">
  <img src="https://github.com/cshernandez9702/mmia_usfq_nlp/blob/main/1.png" alt="Diagrama de Flujo del Proceso RAG" width="600">
</p>


### Generación Aumentada por Recuperación (RAG)

**RAG** es un método que combina modelos basados en recuperación con modelos basados en generación para producir respuestas más precisas y contextualmente relevantes. Recupera documentos relevantes de una base de conocimiento y los utiliza como contexto para generar respuestas a las consultas de los usuarios.

*[Insertar un diagrama de flujo del proceso RAG, mostrando cómo interactúan los componentes de recuperación y generación]*

### Llama 3

**Llama 3** es un modelo de lenguaje grande desarrollado para entender y generar texto similar al humano. Destaca en diversas tareas de PLN y, cuando se afina adecuadamente, puede proporcionar respuestas detalladas y contextualmente precisas.

### Combinación de RAG y Llama 3

Al integrar RAG con Llama 3, mejoramos la capacidad del modelo para generar respuestas que son tanto contextualmente relevantes como informadas por el contenido específico de nuestros PDFs locales. Esta combinación nos permite construir un chatbot capaz de manejar consultas complejas sobre documentos confidenciales sin comprometer la seguridad de los datos.

## Objetivos

- **Desarrollar un chatbot** que pueda consultar documentos PDF locales utilizando RAG y Llama 3.
- **Garantizar la confidencialidad de los datos** procesando los documentos localmente sin enviar datos a servidores externos.
- **Proporcionar respuestas precisas y contextualmente relevantes** a las consultas de los usuarios aprovechando técnicas avanzadas de PLN.
- **Crear una interfaz interactiva** para que los usuarios interactúen con el chatbot de manera fluida.

## Materiales y Herramientas

- **Lenguaje de Programación**: Python
- **Librerías y Frameworks**:
  - `transformers` (Hugging Face Transformers)
  - `langchain`
  - `torch` (PyTorch)
  - `Chroma` (para almacenamiento vectorial)
  - `unstructured` (para procesamiento de PDFs)
- **Modelos**:
  - Llama 3 (8B parámetros)
  - Sentence Transformers para embeddings
- **Datos**:
  - Documentos PDF confidenciales
- **Hardware**:
  - Sistema con capacidad de GPU para inferencia del modelo (opcional pero recomendado)

## Implementación Detallada

### Visión General

El proyecto consta de varios componentes clave:

1. **Carga y Configuración del Modelo**: Carga del modelo Llama 3 y su tokenizador.
2. **Carga y Procesamiento de PDFs**: Lectura de PDFs y extracción de contenido textual.
3. **Generación de Embeddings y Configuración del Vector Store**: Conversión de texto en embeddings y almacenamiento usando Chroma.
4. **Configuración de la Cadena RAG**: Creación de una cadena de generación aumentada por recuperación.
5. **Interfaz del Chatbot**: Construcción de un chatbot interactivo que maneja las consultas de los usuarios.

### Carga y Configuración del Modelo

Comenzamos cargando el modelo Llama 3 y su tokenizador. El modelo se configura para ejecutarse eficientemente en el hardware disponible, utilizando aceleración GPU si es posible.

**Función utilizada**: `load_model_and_tokenizer(model_path)`

- **Cuantización**: Aunque consideramos la cuantización de 8 bits para reducir el uso de memoria, optamos por cargar el modelo completo ya que disponíamos de recursos suficientes.
- **Mapeo de Dispositivos**: El modelo se asigna automáticamente a los dispositivos disponibles (CPU o GPU).

### Carga y Procesamiento de PDFs usando la librería `unstructured`

Para procesar los documentos PDF, utilizamos la librería `unstructured`, que permite extraer el texto de los PDFs sin incluir imágenes. Este paso es crucial para crear embeddings significativos.

**Función utilizada**: `load_pdf_with_unstructured(pdf_path)`

- **Extracción de Texto**: Nos enfocamos en el texto narrativo, texto regular, elementos de lista y tablas.
- **Exclusión de Imágenes**: Se excluyen las imágenes para mantener la confidencialidad y centrarse en la información textual.
- **Procesamiento Detallado**: Clasificamos y combinamos los diferentes tipos de contenido para su posterior procesamiento.

*[Insertar una imagen que muestre el proceso de extracción y clasificación de contenido de los PDFs]*

### Generación de Embeddings y Configuración del Vector Store

Generamos embeddings a partir del texto extraído utilizando un modelo pre-entrenado de Sentence Transformer y los almacenamos usando Chroma para una recuperación eficiente.

**Función utilizada**: `setup_vectorstore_unstructured(doc_text, table_text, model_name)`

- **División de Texto**: El texto se divide en fragmentos manejables para la generación de embeddings.
- **Embeddings**: Los embeddings de alta calidad capturan el significado semántico, crucial para una recuperación precisa.
- **Vector Store**: Chroma almacena los embeddings y facilita la búsqueda de similitud durante la recuperación.

*[Insertar una imagen que represente el proceso de generación de embeddings y su almacenamiento en Chroma]*

### Configuración de la Cadena RAG

Creamos una cadena de generación aumentada por recuperación que utiliza los embeddings para recuperar documentos relevantes y generar respuestas usando el modelo Llama 3.

**Funciones utilizadas**:

- `vectorstore.as_retriever()`
- `RetrievalQA.from_chain_type(llm, chain_type, retriever, verbose)`

- **Retriever**: Recupera documentos similares a la consulta.
- **Cadena LLM**: Genera respuestas utilizando los documentos recuperados como contexto.

### Interfaz del Chatbot

Implementamos una clase de chatbot que mantiene un historial de conversación y maneja las interacciones con el usuario.

**Clase utilizada**: `RAGChatbot`

- **Gestión del Historial**: Mantiene un registro de interacciones previas para proporcionar contexto.
- **Bucle Interactivo**: Solicita continuamente entradas del usuario y genera respuestas.

### Configuraciones Clave

- **Tokens Máximos**: Establecido para asegurar que las respuestas sean concisas y dentro de las limitaciones del modelo.
- **Tamaño y Solapamiento de Fragmentos**: Optimizados para equilibrar el contexto y el rendimiento.
- **Uso de Dispositivos**: Configurado para utilizar GPU si está disponible para una inferencia más rápida.

## Resultados

El chatbot responde exitosamente a consultas recuperando información relevante de los PDFs locales y generando respuestas coherentes.

### Ejemplo de Interacción

**Usuario**: "Explica cuáles son los tipos de transacciones a compensarse por RTC en español."

**Chatbot**: "Las transacciones a compensar por el RTC incluyen cheques, transferencias electrónicas y otras operaciones financieras detalladas en los documentos proporcionados."

**Tiempo Tomado**: 3.2 segundos

*[Insertar una captura de pantalla de la interacción del chatbot demostrando el ejemplo anterior]*

### Métricas de Rendimiento

- **Tiempo de Respuesta**: El tiempo promedio de respuesta es aceptable para uso interactivo.
- **Precisión**: Las respuestas son contextualmente relevantes y precisas basadas en los PDFs proporcionados.

## Limitaciones y Posibles Mejoras

### Limitaciones

- **Requerimientos de Hardware**: El modelo Llama 3 es grande y requiere recursos significativos, lo que puede limitar su implementación en sistemas con hardware modesto.
- **Tiempo de Inferencia**: A pesar de las optimizaciones, el tiempo de respuesta puede ser elevado para consultas muy complejas o documentos muy extensos.
- **Procesamiento de Imágenes**: Actualmente, el sistema no procesa imágenes o gráficos que puedan contener información relevante.
- **Dependencia del Contenido**: La precisión de las respuestas depende de la calidad y relevancia de los documentos PDF cargados.

### Posibles Mejoras

- **Optimización del Modelo**: Implementar técnicas como cuantización o destilación para reducir el tamaño del modelo y mejorar el rendimiento en hardware limitado.
- **Ampliación del Soporte de Documentos**: Extender el soporte a otros formatos de documentos como Word, Excel, etc.
- **Procesamiento de Imágenes**: Integrar capacidades para extraer y procesar texto de imágenes o gráficos incluidos en los PDFs.
- **Mejora de la Interfaz**: Desarrollar una interfaz web o gráfica para mejorar la accesibilidad y la experiencia del usuario.
- **Actualización Dinámica del Conocimiento**: Implementar mecanismos para actualizar el vector store en tiempo real con nuevos documentos o información.

## Conclusión

Hemos desarrollado un chatbot funcional basado en RAG utilizando Llama 3 que puede consultar eficientemente PDFs locales. Este sistema garantiza la confidencialidad de los datos al procesar los documentos localmente y proporciona respuestas precisas y contextualmente relevantes a las consultas de los usuarios. El proyecto demuestra el potencial de combinar técnicas avanzadas de PLN para aplicaciones prácticas en la recuperación de información.

### Trabajo Futuro

- **Integración con Sistemas Empresariales**: Conectar el chatbot con sistemas internos para ampliar su utilidad en entornos corporativos.
- **Mejora de la Personalización**: Permitir ajustes personalizados en el comportamiento del chatbot según las necesidades del usuario.
- **Soporte Multilingüe**: Ampliar la capacidad del chatbot para manejar consultas y documentos en múltiples idiomas.

## Referencias

- [Documentación de Hugging Face Transformers](https://huggingface.co/docs/transformers/index)
- [Documentación de LangChain](https://langchain.readthedocs.io/en/latest/)
- [Documentación de Chroma](https://www.trychroma.com/)
- [Librería unstructured](https://github.com/Unstructured-IO/unstructured)

## Apéndice

### Fragmentos de Código

#### Carga de Múltiples PDFs

**Función utilizada**: `load_and_add_to_vectorstore(pdf_paths)`

*[Insertar un diagrama o flujo que muestre el proceso de carga de múltiples PDFs y actualización del vector store]*

#### Consulta al Sistema RAG

**Función utilizada**: `query_rag_system(qa_chain, query)`

### Ejemplo de Interacción con el Chatbot

*[Insertar capturas de pantalla adicionales del chatbot manejando diferentes consultas, mostrando su capacidad para mantener el contexto y proporcionar respuestas precisas]*

---

**Nota sobre Imágenes y Diagramas**:

- **Diagrama de Flujo del Proceso RAG**: Debe ilustrar cómo el sistema recupera documentos relevantes y genera respuestas.
- **Diagrama de Generación de Embeddings**: Representar cómo el texto se convierte en embeddings y se almacena.
- **Capturas de Interacción del Chatbot**: Capturar interacciones reales para mostrar las capacidades del chatbot.


