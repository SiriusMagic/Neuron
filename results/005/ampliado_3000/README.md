# 005 ampliado — 3000 pasos, 10 semillas

**Comando:** `PYTHONPATH=../Notbook python experiments/language005.py --steps 3000 --seeds 11 23 47 61 73 89 101 113 127 139 --out .../metricas.csv`
**Código:** Neuron `16b8512`, Notbook `77aa0d1`. **Hardware:** CPU Intel Xeon 2,30 GHz, 4 hilos, sin GPU; torch 2.15.0+cpu. ~53 s por corrida.
**Datos:** iguales que la primera ejecución (2500/350/350 reseñas, 0 duplicados exactos entre conjuntos). Cada modelo procesa 3000 × 24 × 80 posiciones (≈29 épocas del conjunto de entrenamiento).

| Modelo | BPC prueba (media ± d.e., n=10) | Δ vs GELU, IC95 pareado | Parámetros | Mejor paso (validación) |
|---|---|---|---|---|
| **gelu** | **2,0197 ± 0,0106** | — | 174 592 | 1500 (las 10) |
| cuadratica (ReLU²) | 2,0222 ± 0,0065 | +0,003 [−0,005, +0,010] | 174 592 | 1500 (las 10) |
| lineal | 2,0424 ± 0,0050 | +0,023 [+0,016, +0,029] | 174 592 | 2250 (las 10) |
| eliminada | 2,0528 ± 0,0137 | +0,033 [+0,021, +0,046] | 141 184 | 2250 (las 10) |

## Lectura

- **Convergencia:** a diferencia de la ejecución de 180 pasos, el mejor paso ya no es el último. Las variantes GELU y ReLU² alcanzan su mejor validación a los 1500 pasos y luego sobreajustan: hay pocos datos (2500 reseñas) para el modelo.
- **El FFN aporta:** quitarlo empeora +0,033 BPC (IC95 excluye 0, 10/10 semillas comparables). Esto revierte la lectura de los 180 pasos, donde no se veía diferencia.
- **La no linealidad aporta:** el FFN lineal empeora +0,023 BPC con los mismos parámetros. La activación no se puede eliminar en este régimen.
- **ReLU² ≈ GELU:** diferencia +0,003, IC95 incluye 0. No hay evidencia de que una sea mejor; ReLU² es algo más barata de calcular, pero aquí no se midió ninguna ventaja de velocidad.
- La latencia (ms por lote de 24) sigue siendo poco fiable a esta escala.

## Limitaciones

- El modelo es diminuto (2 bloques, dim 64) y los datos son pocos. Hubo sobreajuste, así que el resultado describe un régimen limitado por datos.
- La validación solo se evaluó cada 750 pasos; el óptimo real puede estar entre esos puntos.
- `lr` = 0,001 fijo para todas, sin barrido.
- Un solo corpus (reseñas de Amazon).

## Decisión (filosofía: eliminar lo que no aporta)

Se conservan el FFN y una activación no lineal. Lineal y sin FFN quedan descartados como candidatos en este régimen. El siguiente paso es 006: comparar contra SwiGLU, la referencia industrial, y la candidata bilineal.
