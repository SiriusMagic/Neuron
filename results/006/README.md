# 006 — SwiGLU (referencia) vs neurona bilineal (candidata)

**Comando:** `PYTHONPATH=../Notbook python experiments/language006.py --steps 3000 --seeds 11 23 47 61 73 89 101 113 127 139 --out .../metricas.csv`
**Código:** Neuron `938eea4`, Notbook `77aa0d1`. **Hardware:** CPU Intel Xeon 2,30 GHz, 4 hilos; torch 2.15.0+cpu.
**Datos y entrenamiento:** idénticos a 005 ampliado (2500/350/350 reseñas, 0 duplicados exactos, 3000 pasos, lr 0,001). Solo cambia el FFN.

**Hipótesis escrita antes de ejecutar:** la bilineal (SwiGLU sin `silu`) iguala o supera a SwiGLU con los mismos parámetros.

| Modelo | BPC prueba (media ± d.e., n=10) | Δ vs SwiGLU, IC95 pareado | Parámetros | Mejor paso |
|---|---|---|---|---|
| gelu | 2,0218 ± 0,0092 | −0,003 [−0,011, +0,006] | 174 592 | 1500 (10/10) |
| **swiglu** | 2,0243 ± 0,0120 | — | 174 548 | 1500 (10/10) |
| bilineal | 2,0287 ± 0,0108 | +0,004 [−0,005, +0,013] | 174 548 | 1500 (10/10) |
| eliminada | 2,0515 ± 0,0143 | +0,027 [+0,016, +0,039] | 141 184 | 2250 (10/10) |

## Lectura

- **Empate a tres:** GELU, SwiGLU y bilineal no se distinguen; los IC95 incluyen 0. La bilineal ganó a SwiGLU en 5 de 10 semillas.
- **La hipótesis no se confirma ni se refuta:** la bilineal no es superior. Que sea equivalente tampoco está demostrado, porque no se fijó un margen de no inferioridad y el IC llega a +0,013.
- **Sin FFN vuelve a perder** (+0,027), lo que replica 005.
- **Que SwiGLU no supere a GELU indica que el banco es poco sensible** en este régimen: todos alcanzan su mejor punto a los 1500 pasos y después sobreajustan (2500 reseñas, ~29 épocas). En la literatura, la ventaja de SwiGLU aparece en modelos y datos mucho mayores.

## Limitaciones

Las mismas de 005 ampliado: modelo diminuto, pocos datos con sobreajuste, validación cada 750 pasos, lr fijo y un solo corpus. La latencia a esta escala no es fiable.

## Siguiente experimento mínimo (006b)

Repetir 006 sin cambiar la arquitectura, con muchos más documentos de entrenamiento (≈1 época, sin sobreajuste), para ver si el banco puede separar SwiGLU de GELU. Si ni así separa a la referencia industrial del control, hay que escalar el modelo antes de juzgar candidatas.
