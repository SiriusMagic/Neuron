# Resultados 005 — primera ejecución real (exploratoria)

**Fecha:** 2026-11-08. **Comando:** `PYTHONPATH=../Notbook python experiments/language005.py --steps 180 --seeds 11 23 47`
**Código:** Neuron `8c2c5b5`, Notbook `77aa0d1` (solo lectura). **Hardware:** CPU Intel Xeon 2,30 GHz, 4 hilos, sin GPU. torch 2.15.0+cpu, datasets 5.1.0, Python 3.13.
**Datos:** `SetFit/amazon_reviews_multi_es`; 2500 docs de entrenamiento y 350 de validación (del split `train`, sin solape de filas), 350 de prueba (split `test`). 0 duplicados exactos entre conjuntos tras normalizar. Vocabulario silábico de 1600 construido solo con entrenamiento. 143 272 tokens de entrenamiento.
**BPC:** bits por carácter del texto normalizado (minúsculas, solo `a–z áéíóúüñ`, dígitos y `¿¡!?.,;:()-`, separados por un espacio); incluye la predicción de `<eos>`.

Archivos crudos: [metricas.csv](metricas.csv), [metricas.json](metricas.json) (config y entorno), [log_ejecucion.txt](log_ejecucion.txt), [resumen.txt](resumen.txt) (`python experiments/analizar005.py`).

| Modelo | BPC prueba (media ± d.e., 3 semillas) | Δ vs GELU, IC95 pareado | Parámetros |
|---|---|---|---|
| gelu | 2,7833 ± 0,0184 | — | 174 592 |
| lineal | 2,7686 ± 0,0433 | −0,015 [−0,083, +0,054] | 174 592 |
| cuadratica (ReLU²) | 2,7545 ± 0,0115 | −0,029 [−0,046, −0,012] | 174 592 |
| eliminada | 2,7761 ± 0,0186 | −0,007 [−0,096, +0,082] | 141 184 |

## Lectura

**No hay conclusión sobre qué FFN es mejor.**

- **Entrenamiento insuficiente:** las 12 corridas tuvieron su mejor validación en el último paso (180), así que todas seguían mejorando. Con 180 pasos se compara la velocidad inicial de aprendizaje, no la capacidad de cada variante.
- Las diferencias (0,01–0,03 BPC) son del mismo orden que la dispersión entre semillas. Con 3 semillas, el IC de ReLU² que excluye 0 es exploratorio: no se puede declarar ganadora.
- Quitar el FFN (−19 % de parámetros) no empeoró de forma detectable *en este régimen*. Eso no muestra que el FFN sobre; puede indicar que a los 180 pasos el FFN todavía no se aprovecha.
- La latencia (ms por lote de 24, CPU) es ruidosa en esta escala: el modelo sin FFN midió más que GELU, cosa imposible por cómputo. No usarla para concluir.

## Siguiente experimento mínimo

Repetir el mismo 005 sin cambiar la arquitectura, con más pasos (hasta que el mejor paso de validación deje de ser el último) y 10 semillas. Solo con eso tiene sentido decidir si GELU, ReLU², un FFN lineal o ningún FFN merecen seguir.
