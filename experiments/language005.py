"""Notbook 005: primera prueba causal de lenguaje para neuronas simples.
Fuente de sílabas: SiriusMagic/Notbook/laboratorio/silabas.py.
Ejecutar: python experiments/language005.py --steps 180 --seeds 11 23 47
Prueba de funcionamiento: python experiments/language005.py --smoke --steps 3 --seeds 11
"""
import argparse
import csv
import json
import platform
import subprocess
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

# Se carga EXACTAMENTE la implementación existente de Notbook, no una nueva heurística.
from laboratorio.silabas import silabas

PATRON = re.compile(r"[a-záéíóúüñ]+|[0-9]+|[¿¡!?.,;:()\x2d]")
PALABRA = re.compile(r"[a-záéíóúüñ]+$")
CARACTERES = "abcdefghijklmnñopqrstuvwxyzáéíóúü0123456789¿¡!?.,;:()-"
MARCA = "▁"


def normalizar(texto):
    """El BPC solo se calcula sobre este texto normalizado, nunca sobre el original."""
    return " ".join(PATRON.findall(texto.lower()))


def particiones(smoke, n_train, n_val, n_test):
    if smoke:
        sujetos = ["gato", "perro", "niño", "científico", "médico", "maestro", "robot", "pájaro"]
        objetos = ["libro", "árbol", "reloj", "jardín", "puente", "mapa", "pan", "barco"]
        verbos = ["observa", "encuentra", "describe", "revisa", "recuerda", "imagina"]
        textos = []
        for i in range(400):
            a, b, c = sujetos[i % len(sujetos)], objetos[(i * 3 + i // 8) % len(objetos)], verbos[(i // 7) % len(verbos)]
            textos.append(f"El {a} {c} el {b}. Luego explica lo que aprendió.")
            textos.append(f"En la mañana, el {a} piensa en el {b} y conversa.")
        # Solo prueba de ejecución, no constituye un corpus de lenguaje real.
        rng = random.Random(2026)
        rng.shuffle(textos)
        return textos[:120], textos[120:150], textos[150:180], "sintético: solo smoke test"

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Instala datasets con: pip install datasets") from exc
    nombre = "SetFit/amazon_reviews_multi_es"  # fuente ya utilizada en Notbook
    train_raw = load_dataset(nombre, split="train")
    test_raw = load_dataset(nombre, split="test")
    train_raw = train_raw.shuffle(seed=2026)
    test_raw = test_raw.shuffle(seed=2027)

    vistos = set()
    def tomar(ds, desde, n):
        # Devuelve también la siguiente fila sin leer, para que validación empiece
        # exactamente donde terminó entrenamiento (sin solapar filas).
        out = []
        siguiente = desde
        for siguiente in range(desde, len(ds)):
            texto = normalizar(ds[siguiente]["text"])
            if len(texto) < 25 or texto in vistos:
                continue
            vistos.add(texto)
            out.append(texto)
            if len(out) >= n:
                break
        if len(out) < n:
            raise RuntimeError(f"Solo se encontraron {len(out)} textos únicos; se pidieron {n}")
        return out, siguiente + 1
    train, fin_train = tomar(train_raw, 0, n_train)
    valid, _ = tomar(train_raw, fin_train, n_val)
    test, _ = tomar(test_raw, 0, n_test)
    return train, valid, test, nombre


class Tokenizador:
    """Sílabas del repositorio original, con marcador de palabra y respaldo por caracteres.

    El respaldo evita que un <unk> esconda información en el conjunto de prueba.
    Las reglas de construcción del vocabulario solo usan entrenamiento.
    """
    def __init__(self, entrenamiento, max_vocab=1600):
        obligatorios = ["<pad>", "<bos>", "<eos>"]
        for c in CARACTERES:
            for t in (MARCA + c, c):
                if t not in obligatorios:
                    obligatorios.append(t)
        conteo = Counter()
        for texto in entrenamiento:
            conteo.update(self.unidades(texto))
        self.vocab = dict((t, i) for i, t in enumerate(obligatorios))
        for token, _ in conteo.most_common():
            if len(self.vocab) >= max_vocab:
                break
            if token not in self.vocab:
                self.vocab[token] = len(self.vocab)
        self.pad = self.vocab["<pad>"]
        self.bos = self.vocab["<bos>"]
        self.eos = self.vocab["<eos>"]

    @staticmethod
    def unidades(texto):
        tokens = []
        for trozo in normalizar(texto).split():
            partes = silabas(trozo) if PALABRA.fullmatch(trozo) else [trozo]
            tokens.append(MARCA + partes[0])
            tokens.extend(partes[1:])
        return tokens

    def codificar(self, texto):
        salida = []
        for trozo in normalizar(texto).split():
            partes = silabas(trozo) if PALABRA.fullmatch(trozo) else [trozo]
            unidades = [MARCA + partes[0]] + partes[1:]
            if any(t not in self.vocab for t in unidades):
                # Respaldo reversible del trozo completo: primer carácter marca el espacio.
                unidades = [MARCA + trozo[0]] + list(trozo[1:])
            if any(t not in self.vocab for t in unidades):
                raise ValueError(f"Carácter fuera del alfabeto: {trozo!r}")
            salida.extend(self.vocab[t] for t in unidades)
        return [self.bos] + salida + [self.eos]

    def reversible(self, texto):
        inverso = {v: k for k, v in self.vocab.items()}
        piezas = [inverso[i] for i in self.codificar(texto)[1:-1]]
        return "".join(piezas).replace(MARCA, " ").strip() == normalizar(texto)


def convertir(textos, tokenizador, largo):
    xs, ys = [], []
    caracteres = 0
    for texto in textos:
        norm = normalizar(texto)
        ids = tokenizador.codificar(norm)
        caracteres += len(norm)
        for inicio in range(0, len(ids) - 1, largo):
            # Cada token (salvo <bos>) se predice una sola vez. La entrada se recorta
            # para que tenga el mismo largo que su objetivo; antes, un segmento que
            # terminaba justo en el último token dejaba x con 80 e y con 79.
            b = ids[inicio + 1:inicio + largo + 1]
            a = ids[inicio:inicio + len(b)]
            faltan = largo - len(a)
            xs.append(a + [tokenizador.pad] * faltan)
            ys.append(b + [tokenizador.pad] * faltan)
    return torch.tensor(xs, dtype=torch.long), torch.tensor(ys, dtype=torch.long), caracteres


class Bloque(nn.Module):
    def __init__(self, dim, cabezas, tipo, ocultas):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.atn = nn.MultiheadAttention(dim, cabezas, batch_first=True)
        self.tipo = tipo
        if tipo != "eliminada":
            self.ln2 = nn.LayerNorm(dim)
            self.up = nn.Linear(dim, ocultas)
            self.down = nn.Linear(ocultas, dim)

    def forward(self, x):
        n = x.shape[1]
        mascara = torch.ones((n, n), dtype=torch.bool, device=x.device).triu(1)
        z = self.ln1(x)
        x = x + self.atn(z, z, z, attn_mask=mascara, need_weights=False)[0]
        if self.tipo != "eliminada":
            z = self.up(self.ln2(x))
            if self.tipo == "gelu":
                z = F.gelu(z)
            elif self.tipo == "cuadratica":
                z = F.relu(z).square()  # ReLU²: mismo número de pesos que GELU.
            elif self.tipo == "lineal":
                pass  # Control: ¿hace falta una activación no lineal en el FFN?
            else:
                raise ValueError(self.tipo)
            x = x + self.down(z)
        return x


class Modelo(nn.Module):
    def __init__(self, vocab, tipo, dim=64, ocultas=128, largo=80, capas=2):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(largo, dim)
        self.bloques = nn.ModuleList([Bloque(dim, 4, tipo, ocultas) for _ in range(capas)])
        self.norm = nn.LayerNorm(dim)
        self.cabeza = nn.Linear(dim, vocab, bias=False)
        self.cabeza.weight = self.emb.weight  # Tied embeddings: mismo diseño para todos.
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)

    def forward(self, tokens):
        B, T = tokens.shape
        posiciones = torch.arange(T, device=tokens.device)
        x = self.emb(tokens) + self.pos(posiciones)[None, :, :]
        for bloque in self.bloques:
            x = bloque(x)
        return self.cabeza(self.norm(x))


def entorno():
    """Metadatos para reproducir: versiones, hardware y commit de Notbook."""
    import laboratorio
    datos = {"python": platform.python_version(), "torch": torch.__version__,
             "plataforma": platform.platform(), "cpu": platform.processor() or platform.machine(),
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
             "notbook_commit": "", "neuron_commit": ""}
    try:
        import datasets
        datos["datasets"] = datasets.__version__
    except ImportError:
        datos["datasets"] = ""
    for clave, carpeta in (("notbook_commit", Path(laboratorio.__file__).resolve().parent.parent),
                           ("neuron_commit", Path(__file__).resolve().parent.parent)):
        try:
            datos[clave] = subprocess.run(["git", "-C", str(carpeta), "rev-parse", "HEAD"],
                                          capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    return datos


@torch.no_grad()
def latencia_ms(modelo, x, device, repeticiones=20):
    """Mediana de ms por lote de inferencia (forward completo, sin generación)."""
    modelo.eval()
    a = x[:24].to(device)
    tiempos = []
    for i in range(repeticiones + 3):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        modelo(a)
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= 3:  # descartar calentamiento
            tiempos.append((time.perf_counter() - t) * 1000)
    return sorted(tiempos)[len(tiempos) // 2]


@torch.no_grad()
def medir(modelo, x, y, total_caracteres, pad, device, batch=24):
    modelo.eval()
    suma, tokens = 0.0, 0
    for i in range(0, len(x), batch):
        a, b = x[i:i + batch].to(device), y[i:i + batch].to(device)
        logits = modelo(a)
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), b.reshape(-1),
                               ignore_index=pad, reduction="sum")
        suma += loss.item()
        tokens += int((b != pad).sum())
    nats = suma / max(tokens, 1)
    return {"nll_token": nats, "bpc_normalizado": suma / (max(total_caracteres, 1) * math.log(2)),
            "perplejidad": math.exp(min(nats, 100)), "tokens": tokens}


def ejecutar(smoke=False, steps=180, seeds=(11, 23, 47),
             n_train=2500, n_val=350, n_test=350, batch=24, largo=80, lr=0.001, out=None):
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(min(4, torch.get_num_threads()))
    train, valid, test, fuente = particiones(smoke, n_train, n_val, n_test)
    tok = Tokenizador(train, max_vocab=900 if smoke else 1600)
    assert all(tok.reversible(t) for t in train[:25] + valid[:25] + test[:25])
    tx, ty, _ = convertir(train, tok, largo)
    vx, vy, vc = convertir(valid, tok, largo)
    px, py, pc = convertir(test, tok, largo)
    if (ty != tok.pad).sum() < 100:
        raise ValueError("Corpus insuficiente.")
    resumen = {"fuente": fuente, "dispositivo": str(dispositivo), "vocabulario": len(tok.vocab),
               "docs": [len(train), len(valid), len(test)],
               "segmentos_entreno": len(tx), "segmentos_validacion": len(vx),
               "segmentos_prueba": len(px), "tokens_entreno": int((ty != tok.pad).sum()),
               "fugas_exactas_entre_conjuntos": len(set(train) & set(valid)) + len(set(train) & set(test))
               + len(set(valid) & set(test)), "modo_smoke": smoke}
    config = {"steps": steps, "seeds": list(seeds), "n_train": n_train, "n_val": n_val, "n_test": n_test,
              "batch": batch, "largo": largo, "lr": lr, "dim": 64, "ocultas": 128, "capas": 2,
              "cabezas": 4, "optimizador": "AdamW(weight_decay=0.01)", "clip": 1.0,
              "posiciones_por_modelo_incluye_relleno": steps * batch * largo}
    print(resumen)
    filas = []
    for semilla in seeds:
        for tipo in ("gelu", "lineal", "cuadratica", "eliminada"):
            torch.manual_seed(semilla)
            random.seed(semilla)
            modelo = Modelo(len(tok.vocab), tipo, largo=largo).to(dispositivo)
            opt = torch.optim.AdamW(modelo.parameters(), lr=lr, weight_decay=0.01)
            gen = torch.Generator().manual_seed(semilla + 9001)
            mejor, estado, mejor_step = float("inf"), None, 0
            comienzo = time.perf_counter()
            for paso in range(1, steps + 1):
                modelo.train()
                indices = torch.randint(0, len(tx), (batch,), generator=gen)
                a, b = tx[indices].to(dispositivo), ty[indices].to(dispositivo)
                logits = modelo(a)
                perdida = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                          b.reshape(-1), ignore_index=tok.pad)
                opt.zero_grad(set_to_none=True)
                perdida.backward()
                torch.nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
                opt.step()
                if paso % max(1, steps // 4) == 0 or paso == steps:
                    met = medir(modelo, vx, vy, vc, tok.pad, dispositivo)
                    if met["bpc_normalizado"] < mejor:
                        mejor = met["bpc_normalizado"]
                        estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
                        mejor_step = paso
            if dispositivo.type == "cuda":
                torch.cuda.synchronize()
            segundos = time.perf_counter() - comienzo
            modelo.load_state_dict(estado)
            ms = latencia_ms(modelo, px, dispositivo)
            val = medir(modelo, vx, vy, vc, tok.pad, dispositivo)
            prueba = medir(modelo, px, py, pc, tok.pad, dispositivo)
            fila = {
                "fuente": fuente, "smoke": smoke, "semilla": semilla, "modelo": tipo,
                "parametros": sum(p.numel() for p in modelo.parameters()),
                "bytes_fp32": sum(p.numel() * p.element_size() for p in modelo.parameters()),
                "pasos": steps, "mejor_paso": mejor_step, "segundos_entrenar": round(segundos, 3),
                "bpc_validacion": val["bpc_normalizado"],
                "bpc_prueba": prueba["bpc_normalizado"],
                "nll_token_prueba": prueba["nll_token"],
                "perplejidad_prueba": prueba["perplejidad"],
                "tokens_prueba": prueba["tokens"],
                "chars_normalizados_prueba": pc,
                "ms_inferencia_lote24": round(ms, 3),
                "dispositivo": str(dispositivo),
            }
            filas.append(fila)
            print(f"{tipo:11} semilla={semilla} BPC_valid={fila['bpc_validacion']:.3f} "
                  f"BPC_prueba={fila['bpc_prueba']:.3f} params={fila['parametros']}")
    archivo = Path(out) if out else Path("results/005") / ("smoke.csv" if smoke else "metricas.csv")
    archivo.parent.mkdir(parents=True, exist_ok=True)
    with archivo.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(filas[0]))
        writer.writeheader()
        writer.writerows(filas)
    meta = {"config": config, "datos": resumen, "entorno": entorno()}
    (archivo.with_suffix(".json")).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Guardado: {archivo.resolve()} y {archivo.with_suffix('.json').name}")
    if smoke:
        print("AVISO: corpus sintético; NO interpretar métricas como evidencia sobre LLM.")
    return filas


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=180)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--train-docs", type=int, default=2500)
    parser.add_argument("--val-docs", type=int, default=350)
    parser.add_argument("--test-docs", type=int, default=350)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--out", default=None, help="CSV de salida (por defecto results/005/metricas.csv)")
    args = parser.parse_args()
    ejecutar(smoke=args.smoke, steps=args.steps, seeds=args.seeds,
             n_train=args.train_docs, n_val=args.val_docs, n_test=args.test_docs, batch=args.batch, out=args.out)
