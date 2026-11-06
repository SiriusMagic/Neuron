# Neuron — Laboratorio de neuronas para lenguaje

**Objetivo:** investigar unidades neuronales más simples, eficientes y capaces de aprender lenguaje, empezando por **texto español**. Proyecto experimental abierto; una «nueva neurona» debe demostrar mejoras reales frente a referencias actuales.

**Guía para continuar en Codex:** [AGENTS.md](AGENTS.md). Este README describe decisiones, estado y reproducibilidad del proyecto.

**Repositorios**
- [SiriusMagic/Neuron](https://github.com/SiriusMagic/Neuron): experimentos nuevos, datos y documentación.
- [SiriusMagic/Notbook](https://github.com/SiriusMagic/Notbook): laboratorio de referencia y tokenizador silábico original. Se reutiliza sin modificarlo.
- [Tokenizador original](https://github.com/SiriusMagic/Notbook/blob/main/laboratorio/silabas.py) y [resultados previos sobre sílabas](https://github.com/SiriusMagic/Notbook/blob/main/laboratorio/banco/informes/01-tokenizadores-segmentadores-corrector.md).

## 1. Motor de la investigación: filosofía de cinco pasos

Inspirada en el algoritmo de ingeniería de Elon Musk: **«No optimizar algo que quizá no debería existir».**

1. **Cuestionar** supuestos y requisitos. ¿Es necesaria la suma ponderada, la activación, el sesgo, el FFN, la atención, la capa?
2. **Eliminar** cualquier componente prescindible; probar su ausencia experimentalmente.
3. **Simplificar/optimizar** solo lo que haya demostrado aportar valor.
4. **Acelerar** el diseño y la ejecución, una vez justificada su existencia.
5. **Automatizar** búsquedas, pruebas o decisiones únicamente al final.

**Regla:** antes de añadir complejidad, preguntar «¿qué pasa si lo eliminamos?». No confundir nuevo, diferente o más grande con mejor. Registrar fracasos.

## 2. Reglas científicas obligatorias

- **Desde 005, pruebas principales exclusivamente con texto/lenguaje:** predicción causal, generalización lingüística, memoria contextual y generación cuando corresponda. XOR, círculos, lunas y anillos son tareas auxiliares, no pruebas de superioridad para LLM.
- **Español primero.** En 005 usar el tokenizador silábico de Notbook para las cuatro variantes; no comparar tokenizadores y neuronas a la vez. En una investigación separada sí puede contrastarse sílabas con BPE, caracteres o Unigram.
- **Ablación primero:** modelos convencionales y versiones sin operación. Mantener el resto del modelo constante para aislar causalidad.
- **Rigor en los datos:** entrenamiento, validación y prueba independientes; vocabulario ajustado solo con entrenamiento, detección de duplicados, elección de hiperparámetros con validación y evaluación final en prueba.
- **Métricas de lenguaje:** pérdida logarítmica, BPC (indicar si es texto normalizado), perplejidad, generalización; además parámetros, FLOPs, tiempo de entrenamiento, latencia de inferencia, RAM y bytes del modelo cuando estén disponibles.
- **Presupuestos comparables:** mismo corpus, tokenización, lotes, pasos y semillas; diferencias de parámetros/FLOPs y cualquier ventaja debida a un optimizador deben reconocerse y controlarse.
- **Reproducibilidad:** guardar código, configuración, semillas, origen de datos, hardware, versiones, resultados CSV crudos y análisis. 1–3 semillas solo exploran; buscar 10 semillas para confirmar, más dispersión y controles estadísticos.
- **Ningún resultado inventado:** separar hipótesis de mediciones; una prueba de sintaxis no es un entrenamiento, un smoke test sintético no es una evaluación científica, y la clasificación de reseñas no equivale a generación de texto.
- **No afirmar originalidad sin antecedentes.** Consultar artículos y documentación primaria; no Wikipedia.
- **No confundir equivalencia algebraica, resultados empíricos y redondeo numérico.** La compresión exacta en una familia cuadrática no implica una compresión universal.
- **Mantener el coste de búsqueda:** seleccionar una estructura más pequeña puede requerir entrenar muchas candidatas. Informar tanto coste de inferencia como de búsqueda.
- **No implementar nuevas operaciones si una variante más sencilla iguala el resultado.** La neurona mejor es la que demuestre mejor rendimiento relativo a sus costes.

Las instrucciones prácticas para un agente están en [AGENTS.md](AGENTS.md).

## 3. Experimentos y estado

| Cuaderno | Tema | Evidencia y estado |
|---|---|---|
| [001 — Operaciones de la neurona](notebooks/Notbook_001_Operaciones_de_la_neurona.ipynb) | Aditiva vs. interacción multiplicativa | Exploración geométrica. No prueba de calidad LLM. |
| [002 — Quitar antes de optimizar](notebooks/Notbook_002_Quitar_antes_de_optimizar.ipynb) | Ablación y compilación 28 → 6 | Compresión algebraica específica de una función cuadrática sin SiLU. |
| [003 — Aprender vs. representar](notebooks/Notbook_003_Aprender_vs_representar.ipynb) | Comparación de optimización y complejidad | Cuadrática pequeña con L-BFGS igualó aproximadamente a la representación grande en pruebas sintéticas. |
| [004 — Selección de operaciones](notebooks/Notbook_004_Seleccion_de_operaciones.ipynb) | Reducción de términos cuadráticos | Selección de 2–4 coeficientes; búsqueda de muchos candidatos. |
| **[005 — Lenguaje](notebooks/Notbook_005_Lenguaje.ipynb)** | Ablación de FFN en texto español | **Código publicado; falta ejecutar evaluación real y registrar resultados.** |

Datos históricos: [002](results/002/), [003](results/003/) y [004](results/004/). No reinterpretarlos como prueba de LLM.

### Hallazgos preliminares (solo matemáticos)

- 002–003: la variante sin SiLU de 28 parámetros puede representarse como una cuadrática de seis en dos entradas; **no** aplica automáticamente a la variante con SiLU ni a redes generales.
- 003: al entrenar correctamente con L-BFGS, la cuadrática de seis parámetros obtuvo rendimiento similar a la factorizada de 28 en varias tareas.
- 004: eliminar términos dio modelos más pequeños, pero el procedimiento de selección fue costoso; no se ha demostrado mejora del coste total.

**No hay todavía una nueva neurona demostrada superior para LLM.**

## 4. Notbook 005 — Experimento actual

**Pregunta central:** ¿un Transformer pequeño necesita su bloque de alimentación (*FFN*) y su activación no lineal para aprender lenguaje?

En [experiments/language005.py](experiments/language005.py) se usan cuatro variantes sobre un Transformer causal de dos bloques, con idéntica tokenización y atención:

| Variante | Cambio | Pregunta |
|---|---|---|
| `gelu` | FFN con GELU | Control convencional |
| `lineal` | FFN sin activación | ¿Es necesaria la no linealidad? |
| `cuadratica` | FFN con ReLU² | ¿Puede simplificarse/sustituirse GELU? |
| `eliminada` | Sin FFN | ¿Es necesaria la FFN? |

**Corpus:** `SetFit/amazon_reviews_multi_es`, reseñas reales de español. Se utiliza la entrada de texto, **no** las etiquetas de clasificación.  
**Tokenizador:** función `silabas` de `Notbook/laboratorio/silabas.py`. Se añaden marcadores y respaldo por caracteres para evitar pérdidas silenciosas en palabras no vistas.  
**Métrica principal:** bits por carácter del **texto normalizado**, con explicación de sus límites; el entrenamiento es de predicción del siguiente token.  
**Estado:** existe cuaderno y código; el CI solo verificó sintaxis/JSON. **El entrenamiento sobre los textos reales está pendiente.**

No confundir una diferencia de precisión en clasificación con mayor capacidad generativa. ReLU² no es una invención original; se investiga su utilidad en este sistema concreto.

### Ejecutar en Google Colab desde Android

[**Abrir Notbook 005 en Colab**](https://colab.research.google.com/github/SiriusMagic/Neuron/blob/main/notebooks/Notbook_005_Lenguaje.ipynb)

1. Abrir enlace, seleccionar GPU si está disponible.
2. Ejecutar todas las celdas.
3. Comprobar que el modo `--smoke` pasa. **No presentar sus métricas como hallazgo**.
4. Ejecutar evaluación real (`RUN_FULL=True`) y guardar `results/005/metricas.csv`.
5. Publicar CSV, hardware y conclusiones en Neuron.

### Ejecutar desde Codex / terminal Linux

```bash
git clone https://github.com/SiriusMagic/Neuron.git
git clone https://github.com/SiriusMagic/Notbook.git
cd Neuron
python -m pip install torch datasets
PYTHONPATH=../Notbook python experiments/language005.py --smoke --steps 3 --seeds 11
PYTHONPATH=../Notbook python experiments/language005.py --steps 180 --seeds 11 23 47
```

Estas instrucciones requieren acceso a Internet para obtener los datos. Para interpretar ganancias, ampliar semillas y comparar coste de cómputo/latencia.

## 5. Próximo paso para Codex

**No inventar una nueva arquitectura todavía.** Primero ejecutar 005, solucionar únicamente errores reproducibles, registrar resultados reales y determinar si hay evidencia suficiente para conservar GELU, ReLU², una FFN lineal o ninguna FFN.

Luego formular un **Notbook 006** con la operación mínima que merezca probarse en lenguaje. Si los resultados son inconclusos, ampliar la evaluación en lugar de añadir complejidad.

**Instrucciones específicas para el agente:** [AGENTS.md](AGENTS.md).
