# AGENTS.md — Instrucciones para Codex en Neuron

## Misión
Investigar, desde primeros principios, unidades y bloques neuronales más eficientes **para modelar y generar lenguaje en español**. No asumir que una operación existente es necesaria. No declarar novedades, superioridad, equivalencia exacta ni resultados sin pruebas.

**Meta:** crear una neurona/bloque **superior** a los actuales (GELU y SwiGLU) para lenguaje español. «Superior» solo se afirma si se cumple [CRITERIO_SUPERIOR.md](CRITERIO_SUPERIOR.md); antes es «candidata».

## Filosofía obligatoria (orden estricto)
1. **Cuestionar** cada requisito y componente: cuál es su función y evidencia.
2. **Eliminar** la operación o pieza si es posible; ejecutar ablaciones.
3. **Simplificar/optimizar** solo lo indispensable.
4. **Acelerar** ciclos e inferencia después de validar.
5. **Automatizar** solo al final.
Regla: «No optimizar algo que quizá no debería existir».

## Reglas de evaluación desde Notbook 005
- **Pruebas principales siempre en texto/lenguaje**, no XOR, lunas, círculos o anillos. Esos problemas solo sirven como pruebas auxiliares, nunca como evidencia de superioridad para LLM.
- Español primero. Reutilizar el tokenizador real de `SiriusMagic/Notbook`, en `laboratorio/silabas.py`. **No reescribirlo ni modificar Notbook sin autorización**.
- Mantener un **Transformer causal estándar** como control, y variar solo una pieza por experimento. Incluir ablación con esa pieza eliminada.
- Primer contraste en 005: FFN GELU, FFN lineal, FFN ReLU² y sin FFN, con los mismos datos, particiones, tokenizador, atención y entrenamiento. La variante sin FFN tiene menos parámetros: informar diferencia.
- Distinguir **modelado causal de texto** (pérdida y bits por carácter) de **clasificación de reseñas**, generación abierta o razonamiento. No inferir comprensión a partir de una prueba de predicción.
- Separar entrenamiento/validación/prueba; ajustar vocabulario y cualquier normalización únicamente con entrenamiento; elegir modelos en validación; evaluar prueba al final; buscar y eliminar duplicados o fugas entre conjuntos.
- Medir BPC sobre texto normalizado (especificar qué incluye), pérdida por token, perplejidad, parámetros, tamaño en bytes, FLOPs, velocidad de entrenamiento e inferencia y memoria cuando sea posible.
- Controlar igualdad de tokens procesados y comparar también contra presupuestos de parámetros/FLOPs similares. Reportar diferencias imposibles de igualar. No atribuir al diseño diferencias causadas por optimizador, más épocas o distinta tokenización.
- Pruebas rápidas de 1–3 semillas son exploratorias; **usar idealmente ≥10 semillas** para afirmaciones sólidas. Reportar dispersión/intervalos y pruebas apropiadas; no publicar «ganador» por variación pequeña.
- Los datos **sintéticos de --smoke solo verifican que el código funcione**, no prueban que una neurona aprenda mejor lenguaje.
- Todo resultado debe conservar configuración, semilla, fuente, dispositivo, versiones y datos crudos. Separar métricas ejecutadas de predicciones/hipótesis.
- Publicar avances en archivos reproducibles; no inventar CSV ni completar resultados pendientes. Mantener intactos los históricos 001–004 y sus limitaciones.
- Si una operación no demuestra valor, **eliminarla**. Añadir nuevas operaciones solo tras justificar por qué son necesarias.
- Investigar literatura primaria para comprobar antecedentes, sin Wikipedia.

## Entorno de trabajo
Repositorios separados:
- https://github.com/SiriusMagic/Neuron : **repositorio a editar**.
- https://github.com/SiriusMagic/Notbook : código de tokenización y banco de referencia; **usar como dependencia de solo lectura**.

Desde la carpeta contenedora en Linux/Colab:
```bash
git clone https://github.com/SiriusMagic/Neuron.git
git clone https://github.com/SiriusMagic/Notbook.git
cd Neuron
python -m pip install torch datasets
PYTHONPATH=../Notbook python experiments/language005.py --smoke --steps 3 --seeds 11
PYTHONPATH=../Notbook python experiments/language005.py --steps 180 --seeds 11 23 47
```
El último comando descarga datos reales y puede requerir GPU o más tiempo. **No afirmar que se ejecutó sin evidencia.** Al finalizar, revisar `results/005/metricas.csv` y publicar resultados verificables, metodología y limitaciones.

Para Android: abrir `notebooks/Notbook_005_Lenguaje.ipynb` en Google Colab.

## Antes de comenzar a modificar código
1. Leer `README.md`, `experiments/language005.py` y el cuaderno 005.
2. Revisar `Notbook/laboratorio/silabas.py` y el informe de tokenizadores del laboratorio original.
3. Ejecutar la comprobación rápida; no ajustar la arquitectura hasta tener un control confiable.
4. Ejecutar (o preparar para que se ejecute) el experimento real; guardar datos y hardware.
5. Revisar resultados antes de ampliar la arquitectura; proponer **el experimento mínimo siguiente**.

## Estado
001–004: matemáticas sintéticas, no pruebas de LLM. 005: código publicado; **la prueba sobre datos reales aún está pendiente** al redactar este archivo. GitHub Actions valida sintaxis, no rendimiento.

## Conducta de agente
Preferir cambios pequeños y revisables, pruebas y reportes precisos. Evitar dependencias, complejidad y automatización prematuras. No modificar licencias ni alterar datos históricos. Nunca declarar un entrenamiento completado si solo pasó una prueba de sintaxis.
