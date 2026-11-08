"""Notbook 006: ¿la neurona bilineal (cuadrática, sin activación) iguala o supera a SwiGLU?

Misma tubería que 005 (datos, tokenizador silábico de Notbook, entrenamiento, métricas).
Solo cambia el FFN. Parámetros igualados a ±1 % (ver `parametros` en el CSV):
  gelu      : down(gelu(up(x)))                 oculta 128, control histórico de 005
  swiglu    : down(silu(W1 x) * W2 x)           oculta 85, referencia de LLM abiertos
  bilineal  : down((W1 x) * (W2 x))             oculta 85, candidata: SwiGLU sin activación
  eliminada : sin FFN                           ablación, menos parámetros
Referencias: Shazeer 2020 (arXiv:2002.05202) define GLU, SwiGLU y Bilinear; aquí se mide en español.

Prueba de funcionamiento: PYTHONPATH=../Notbook python experiments/language006.py --smoke --steps 3 --seeds 11
"""
import argparse
import sys
from pathlib import Path

import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import language005 as base  # noqa: E402

OCULTAS = {"gelu": 128, "swiglu": 85, "bilineal": 85}


class FFN(nn.Module):
    def __init__(self, dim, tipo):
        super().__init__()
        self.tipo, h = tipo, OCULTAS[tipo]
        self.norm = nn.LayerNorm(dim)
        self.w1 = nn.Linear(dim, h)
        self.w2 = nn.Linear(dim, h) if tipo != "gelu" else None
        self.down = nn.Linear(h, dim)

    def forward(self, x):
        z = self.norm(x)
        if self.tipo == "gelu":
            h = F.gelu(self.w1(z))
        elif self.tipo == "swiglu":
            h = F.silu(self.w1(z)) * self.w2(z)
        elif self.tipo == "bilineal":
            h = self.w1(z) * self.w2(z)  # interacción multiplicativa pura (cf. 001–004)
        else:
            raise ValueError(self.tipo)
        return self.down(h)


class Bloque(nn.Module):
    def __init__(self, dim, cabezas, tipo):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.atn = nn.MultiheadAttention(dim, cabezas, batch_first=True)
        self.ffn = None if tipo == "eliminada" else FFN(dim, tipo)

    def forward(self, x):
        n = x.shape[1]
        mascara = base.torch.ones((n, n), dtype=base.torch.bool, device=x.device).triu(1)
        z = self.ln1(x)
        x = x + self.atn(z, z, z, attn_mask=mascara, need_weights=False)[0]
        return x if self.ffn is None else x + self.ffn(x)


def crear(vocab, tipo, largo, dim=64, capas=2):
    # Embeddings, posiciones, norma final y cabeza compartida idénticos a 005.
    modelo = base.Modelo(vocab, "eliminada", dim=dim, largo=largo, capas=capas)
    modelo.bloques = nn.ModuleList([Bloque(dim, 4, tipo) for _ in range(capas)])
    return modelo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 47])
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--variantes", nargs="+", default=["gelu", "swiglu", "bilineal", "eliminada"])
    parser.add_argument("--train-docs", type=int, default=2500)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    base.ejecutar(smoke=args.smoke, steps=args.steps, seeds=args.seeds, lr=args.lr, out=args.out,
                  n_train=args.train_docs, batch=args.batch,
                  variantes=args.variantes, crear=crear, experimento="006")
