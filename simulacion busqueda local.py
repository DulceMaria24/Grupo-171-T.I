"""
Simulación animada de Hill Climbing y Simulated Annealing sobre el TSP (25 ciudades).
Reutiliza las funciones de busqueda_local.py y genera, en resultados/:
  - simulacion_tsp.gif        animación (ruta actual y curva de costo de ambos algoritmos)
  - simulacion_cuadros.png    cuadros fijos (inicio, 25 %, 60 % y final) para el informe

Uso:  python simulacion.py
"""
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from busqueda_local import (SEMILLA, N_CIUDADES, OUT, aplicar_2opt, costo, delta_2opt,
                            generar_ciudades, matriz_distancias, vecino_aleatorio)

N_FRAMES = 60
CHECK_SA = 1000          # cada cuántas evaluaciones se guarda una instantánea de SA


def trazar_hc(D, rng):
    """Hill Climbing steepest-ascent: guarda la ruta tras cada movimiento aceptado."""
    n = len(D)
    ruta = list(range(n))
    rng.shuffle(ruta)
    c = costo(ruta, D)
    snaps = [(ruta[:], c, None)]
    while True:
        mejor_d, mi, mj = 0.0, None, None
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                d = delta_2opt(ruta, i, j, D)
                if d < mejor_d - 1e-12:
                    mejor_d, mi, mj = d, i, j
        if mi is None:
            break
        aplicar_2opt(ruta, mi, mj)
        c += mejor_d
        snaps.append((ruta[:], c, None))
    return snaps


def trazar_sa(D, rng, T0=100.0, alpha=0.995, T_min=1e-3, iter_por_T=50):
    """Simulated Annealing: guarda ruta actual, costo actual y T cada CHECK_SA evaluaciones."""
    n = len(D)
    ruta = list(range(n))
    rng.shuffle(ruta)
    c = costo(ruta, D)
    snaps = [(ruta[:], c, T0)]
    T, evals = T0, 0
    while T > T_min:
        for _ in range(iter_por_T):
            i, j, d = vecino_aleatorio(ruta, D, rng)
            evals += 1
            if d < 0 or rng.random() < math.exp(-d / T):
                aplicar_2opt(ruta, i, j)
                c += d
            if evals % CHECK_SA == 0:
                snaps.append((ruta[:], c, T))
        T *= alpha
    snaps.append((ruta[:], c, T))
    return snaps


def indice(k, total, n_frames=N_FRAMES):
    return int(round(k / (n_frames - 1) * (total - 1)))


def dibujar_ruta(ax, ciudades, ruta, titulo):
    ax.clear()
    p = ciudades[ruta + [ruta[0]]]
    ax.plot(p[:, 0], p[:, 1], "-", color="black", lw=1.3, zorder=1)
    ax.scatter(ciudades[:, 0], ciudades[:, 1], c="white", edgecolors="black", s=40, zorder=2)
    ax.scatter(*ciudades[ruta[0]], c="black", s=65, zorder=3)
    ax.set_title(titulo, fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlim(-5, 105)
    ax.set_ylim(-5, 105)
    ax.tick_params(labelsize=6)


def main():
    ciudades = generar_ciudades(N_CIUDADES)
    D = matriz_distancias(ciudades)
    hc = trazar_hc(D, random.Random(SEMILLA))
    sa = trazar_sa(D, random.Random(SEMILLA))
    c_hc = [s[1] for s in hc]
    c_sa = [s[1] for s in sa]
    print(f"HC: {len(hc)} instantáneas, costo final {c_hc[-1]:.2f}")
    print(f"SA: {len(sa)} instantáneas, costo final {c_sa[-1]:.2f}")

    def titulo_hc(k):
        i = indice(k, len(hc))
        return f"Hill Climbing · movimiento {i} · costo {hc[i][1]:.1f}"

    def titulo_sa(k):
        i = indice(k, len(sa))
        return f"Simulated Annealing · T = {sa[i][2]:.2f} · costo {sa[i][1]:.1f}"

    # ------------------------------------------------ GIF
    fig, axs = plt.subplots(2, 2, figsize=(9, 7.2), gridspec_kw={"height_ratios": [3, 1.6]})
    ax_hc, ax_sa, ax_c1, ax_c2 = axs[0, 0], axs[0, 1], axs[1, 0], axs[1, 1]
    lim_y = (min(c_sa + c_hc) - 30, max(c_sa + c_hc) + 30)

    def actualizar(k):
        ih, is_ = indice(k, len(hc)), indice(k, len(sa))
        dibujar_ruta(ax_hc, ciudades, hc[ih][0], titulo_hc(k))
        dibujar_ruta(ax_sa, ciudades, sa[is_][0], titulo_sa(k))
        for ax, costos, i, nom, xl in ((ax_c1, c_hc, ih, "Hill Climbing", "Movimiento aceptado"),
                                       (ax_c2, c_sa, is_, "Simulated Annealing", f"Instantánea (cada {CHECK_SA} evaluaciones)")):
            ax.clear()
            ax.plot(range(len(costos)), costos, color="0.75", lw=1)
            ax.plot(range(i + 1), costos[:i + 1], color="black", lw=1.6)
            ax.set_xlim(0, len(costos) - 1)
            ax.set_ylim(*lim_y)
            ax.set_xlabel(xl, fontsize=7)
            ax.set_ylabel("Costo", fontsize=7)
            ax.grid(alpha=.3)
            ax.tick_params(labelsize=6)
        fig.tight_layout()

    anim = FuncAnimation(fig, actualizar, frames=N_FRAMES)
    anim.save(f"{OUT}/simulacion_tsp.gif", writer=PillowWriter(fps=8))
    plt.close(fig)

    # ------------------------------------------------ cuadros fijos para el PDF
    pasos = [0, 0.25, 0.6, 1.0]
    fig, axs = plt.subplots(2, 4, figsize=(12, 6.2))
    for col, f in enumerate(pasos):
        k = int(round(f * (N_FRAMES - 1)))
        ih, is_ = indice(k, len(hc)), indice(k, len(sa))
        dibujar_ruta(axs[0, col], ciudades, hc[ih][0], f"HC · mov. {ih}\ncosto {hc[ih][1]:.1f}")
        dibujar_ruta(axs[1, col], ciudades, sa[is_][0], f"SA · T = {sa[is_][2]:.2f}\ncosto {sa[is_][1]:.1f}")
    plt.tight_layout()
    plt.savefig(f"{OUT}/simulacion_cuadros.png", dpi=160)
    plt.close()


if __name__ == "__main__":
    main()
