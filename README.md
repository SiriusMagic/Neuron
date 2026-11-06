# Neuron — Laboratorio de investigación neuronal

Proyecto experimental: cuestionar requisitos, eliminar componentes innecesarios, simplificar, acelerar y solo después automatizar.

## Cuadernos
- [001 · Operaciones de la neurona](notebooks/Notbook_001_Operaciones_de_la_neurona.ipynb): neurona aditiva e interactiva.
- [002 · Quitar antes de optimizar](notebooks/Notbook_002_Quitar_antes_de_optimizar.ipynb): ablaciones y compactación algebraica (28 → 6 coeficientes).
- [003 · Aprender frente a representar](notebooks/Notbook_003_Aprender_vs_representar.ipynb): comparar parametrización y optimización, con L-BFGS.
- [004 · Selección de operaciones](notebooks/Notbook_004_Seleccion_de_operaciones.ipynb): diez semillas, cinco tareas, 32 subconjuntos; datos en [results/004](results/004).

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
