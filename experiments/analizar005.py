"""Resumen de results/005/metricas.csv: media, desviación y diferencia pareada contra GELU.

Solo resume datos ya medidos; no ejecuta entrenamientos ni completa resultados.
Uso: python experiments/analizar005.py [ruta.csv]
"""
import csv
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def main(ruta):
    filas = list(csv.DictReader(Path(ruta).open(encoding="utf-8")))
    por = defaultdict(dict)
    for f in filas:
        por[f["modelo"]][int(f["semilla"])] = f
    print(f"Archivo: {ruta}  smoke={filas[0]['smoke']}  semillas={sorted(por['gelu'])}")
    print(f"{'modelo':11} {'n':>2} {'BPC_prueba':>16} {'Δ vs GELU (IC95 pareado)':>30} {'params':>7} {'ms/lote':>8}")
    for modelo, sem in por.items():
        bpc = [float(f["bpc_prueba"]) for f in sem.values()]
        sd = st.stdev(bpc) if len(bpc) > 1 else float("nan")
        comunes = sorted(set(sem) & set(por["gelu"]))
        d = [float(sem[s]["bpc_prueba"]) - float(por["gelu"][s]["bpc_prueba"]) for s in comunes]
        if modelo == "gelu":
            delta = "—"
        elif len(d) > 1:
            m, e = st.mean(d), st.stdev(d) / math.sqrt(len(d))
            t = T95.get(len(d) - 1, 2.0)
            delta = f"{m:+.4f} [{m - t * e:+.4f}, {m + t * e:+.4f}]"
        else:
            delta = f"{d[0]:+.4f} (1 semilla)"
        ms = st.median(float(f.get("ms_inferencia_lote24") or "nan") for f in sem.values())
        print(f"{modelo:11} {len(bpc):>2} {st.mean(bpc):>9.4f} ± {sd:.4f} {delta:>30} "
              f"{sem[min(sem)]['parametros']:>7} {ms:>8.2f}")
    print("Δ<0 significa menor BPC que GELU. Con <10 semillas el resultado es exploratorio.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/005/metricas.csv")
