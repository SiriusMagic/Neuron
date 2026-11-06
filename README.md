# Neuron — Laboratorio de investigación neuronal

Proyecto experimental: cuestionar requisitos, eliminar componentes innecesarios, simplificar, acelerar y solo después automatizar.

## Cuadernos
- [001 · Operaciones de la neurona](notebooks/Notbook_001_Operaciones_de_la_neurona.ipynb): neurona aditiva e interactiva.
- [002 · Quitar antes de optimizar](notebooks/Notbook_002_Quitar_antes_de_optimizar.ipynb): ablaciones y compactación algebraica (28 → 6 coeficientes).
- [003 · Aprender frente a representar](notebooks/Notbook_003_Aprender_vs_representar.ipynb): comparar parametrización y optimización, con L-BFGS.
- [004 · Selección de operaciones](notebooks/Notbook_004_Seleccion_de_operaciones.ipynb): diez semillas, cinco tareas, 32 subconjuntos; datos en [results/004](results/004).


## Desde 005: evaluación de lenguaje

- [005 · Ablación de operaciones para lenguaje causal](notebooks/Notbook_005_Lenguaje.ipynb).
- [Implementación reproducible](experiments/language005.py).
- [Ejecutar 005 en Google Colab](https://colab.research.google.com/github/SiriusMagic/Neuron/blob/main/notebooks/Notbook_005_Lenguaje.ipynb).

**Cambio de protocolo:** 001–004 son exploración matemática y NO evidencia de mejora para LLM. Desde 005 las evaluaciones principales usan texto. Comparación inicial: Transformer causal con GELU, ReLU² o sin FFN. La atención, el tokenizador silábico original de [Notbook](https://github.com/SiriusMagic/Notbook/blob/main/laboratorio/silabas.py), la partición de datos y el procedimiento de entrenamiento se mantienen constantes.

Corpus real: reseñas en español `SetFit/amazon_reviews_multi_es`, ya utilizadas en Notbook. El vocabulario se construye solo con entrenamiento; validación y prueba se mantienen separadas. Métrica: bits por carácter **del texto normalizado**, que incluye la codificación de fin de documento. Se evita contar tokens desconocidos como si no contuvieran información mediante respaldo por caracteres.

**Estado:** protocolo y código publicados, resultados de la evaluación real pendientes de ejecución en Colab. El modo `--smoke` solo verifica funcionamiento y NO valida la hipótesis. Las métricas se guardan en `results/005/` al ejecutar. No extrapolar experimentos pequeños a LLM de escala industrial.

## Resultados e interpretación
Las redes sin activación SiLU estudiadas en 002–003 pueden representar la misma función cuadrática de dos variables y compilarse algebraicamente a seis coeficientes. Esta equivalencia **no se aplica** a la variante con SiLU ni a redes arbitrarias.

En 003, la cuadrática de seis parámetros entrenada con L-BFGS igualó aproximadamente el rendimiento de la representación factorizada de 28 en las tareas sintéticas. Esto cuestiona la ventaja aparente de la parametrización más grande.

En 004, la selección por validación redujo a 2–4 parámetros (incluido sesgo) en las cinco tareas sin degradación significativa de precisión. Sin embargo, se entrenan 32 modelos por tarea y semilla; **no demuestra ahorro de coste de entrenamiento**. Tampoco hay equivalencia exacta entre todos los modelos seleccionados y los completos.

## Datos
- `results/002/`: entrenamientos y compilación.
- `results/003/`: optimización, resultados y equivalencia.
- `results/004/`: experimentos, elecciones y resumen.

## Alcance
Resultados exploratorios en problemas artificiales bidimensionales, no pruebas de superioridad general, lenguaje o hardware móvil. Las diferencias de FP32/FP64 y el dominio evaluado importan. El objetivo es compartir código y datos reproducibles.

## Ejecución
Abrir los archivos `.ipynb` en Google Colab y ejecutar las celdas en orden. Bibliotecas: Python, PyTorch, NumPy, SciPy, pandas, scikit-learn y matplotlib.
