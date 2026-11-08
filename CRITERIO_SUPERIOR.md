# Objetivo: una neurona superior — y cómo se demuestra

**Meta del proyecto:** diseñar una unidad o bloque neuronal **superior** a los actuales para modelar español: mejor predicción de texto con el mismo coste, o el mismo resultado con menos coste.

«Superior» no se declara: se demuestra con los criterios de abajo. Mientras no se cumplan todos, la variante es **candidata**, no superior.

## 1. Contra quién se compara

- **GELU:** control histórico del proyecto (005).
- **SwiGLU** (Shazeer, 2020, *GLU Variants Improve Transformer*, arXiv:2002.05202): el FFN de los LLM grandes abiertos actuales. **Es la referencia principal a vencer**; superar solo a GELU no basta.
- **Ablación:** el mismo modelo sin la pieza nueva.

### Modelos cerrados (GPT, Claude, Gemini…)

La meta es superar a las neuronas de los LLM más grandes de la industria. Pero la arquitectura interna de los modelos cerrados **no es pública**, así que no se puede comparar contra ella ni afirmar que se la supera. Se usa como sustituto **lo mejor documentado públicamente** en LLM grandes abiertos:

- FFN **SwiGLU** (LLaMA, Qwen, Mistral, DeepSeek y otros, según sus informes técnicos).
- A gran escala, **mezcla de expertos (MoE)** con expertos SwiGLU. Solo es comparable cuando el proyecto llegue a esa escala.

Ganar al FFN de un modelo pequeño **no** equivale a ganar a GPT o Claude como sistema: esos modelos difieren en datos, escala, entrenamiento y alineamiento. La afirmación válida es: «esta neurona supera a SwiGLU a igual coste en N tamaños de modelo».

### Multimodal (futuro)

Primero texto en español. Una neurona que gane en texto se probará después en otras modalidades (imagen, audio) contra las referencias públicas de cada una. La ventaja en texto no se extrapola sin medir.

## 2. Qué significa «superior» (al menos una de estas, sin empeorar las otras)

| Tipo | Condición |
|---|---|
| **Más precisa** | Menor BPC de prueba que la mejor referencia, con parámetros y FLOPs iguales (±2 %) y los mismos tokens. |
| **Más rápida / barata** | BPC no peor que la mejor referencia (margen de no inferioridad fijado *antes* de correr, p. ej. +0,01 BPC), con ≥15 % menos FLOPs, latencia medida o memoria. |
| **Más eficiente en datos** | Alcanza el BPC de la referencia con ≥20 % menos tokens de entrenamiento. |

## 3. Condiciones obligatorias para aceptar

1. **≥10 semillas**, con diferencia pareada contra la referencia e IC95 que excluya 0 (o que quede dentro del margen, si es de no inferioridad).
2. **Convergencia:** el mejor paso de validación no puede ser el último. Si lo es, se entrena más.
3. **Ajuste justo:** el mismo barrido de tasa de aprendizaje (p. ej. 3 valores) para cada variante, elegido en validación. Así no se gana por un `lr` mejor elegido.
4. **Escala:** la ventaja debe mantenerse en al menos **3 tamaños** de modelo crecientes. Muchas mejoras pequeñas desaparecen al escalar.
5. **Prueba final una sola vez:** la hipótesis y el margen se escriben antes de ejecutar; el conjunto de prueba solo se mira al final.
6. **Coste real:** la latencia se mide en hardware declarado, con suficientes repeticiones y tamaños donde el cómputo domine. La que da 005 a esta escala es ruido.
7. **Antecedentes:** buscar en literatura primaria si la operación ya existe. Si existe, el aporte es la evidencia en español, no la invención.

## 4. Qué NO prueba superioridad

- Pruebas `--smoke`, XOR, lunas, círculos o anillos.
- 1–3 semillas, o diferencias del tamaño de la dispersión entre semillas.
- Mejor BPC conseguido con más parámetros, más pasos u otro optimizador.
- Menor BPC ≠ más «inteligente». Comprensión, razonamiento y generación necesitan evaluaciones propias.

## 5. Ruta

1. **005 ampliado:** GELU / lineal / ReLU² / sin FFN, con más pasos y 10 semillas → saber qué pieza aporta.
2. **006:** añadir SwiGLU como referencia y la neurona **bilineal** `(W₁x)⊙(W₂x)` (cuadrática, enlaza con 001–004), con parámetros igualados.
3. **007:** la mejor candidata en 3 tamaños, con barrido de `lr` → criterio completo.
4. Solo si pasa todo: optimizar velocidad (compilar, fusionar operaciones) y automatizar la búsqueda.

Si una candidata falla, se registra el fracaso y se elimina; no se reajusta hasta que «gane».
