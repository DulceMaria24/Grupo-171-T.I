"""
Sección 3 - Búsqueda Local: Hill Climbing y Simulated Annealing
Fundamentos de Inteligencia Artificial - UPA - Grupo 171

Problema 1 (principal): TSP (ruta mínima) con 25 ciudades, vecindad 2-opt.
Problema 2 (complementario): función Rastrigin 2D, con paso de vecindad eta (η).

Uso:  python busqueda_local.py
Salida: carpeta resultados/ con gráficas (PNG) y resultados.json
"""
import json
import math
import os
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEMILLA = 171
N_CIUDADES = 25
N_CORRIDAS = 30
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultados")
os.makedirs(OUT, exist_ok=True)

AZUL, VERDE, GRIS = "black", "0.35", "0.6"


# ---------------------------------------------------------------- TSP
def generar_ciudades(n, semilla=SEMILLA):
    rng = np.random.default_rng(semilla)
    return rng.uniform(0, 100, size=(n, 2))


def matriz_distancias(c):
    d = c[:, None, :] - c[None, :, :]
    return np.sqrt((d ** 2).sum(-1))


def costo(ruta, D):
    """Función objetivo: longitud total del ciclo hamiltoniano."""
    return sum(D[ruta[i], ruta[(i + 1) % len(ruta)]] for i in range(len(ruta)))


def delta_2opt(ruta, i, j, D):
    """Cambio de costo al invertir el segmento ruta[i..j] (O(1))."""
    n = len(ruta)
    a, b = ruta[i - 1], ruta[i]
    c, d = ruta[j], ruta[(j + 1) % n]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def vecino_aleatorio(ruta, D, rng):
    n = len(ruta)
    i, j = sorted(rng.sample(range(1, n), 2))
    return i, j, delta_2opt(ruta, i, j, D)


def aplicar_2opt(ruta, i, j):
    ruta[i:j + 1] = reversed(ruta[i:j + 1])


def hill_climbing_tsp(D, rng, max_iter=20000, reinicios=1):
    """Hill Climbing de ascenso más pronunciado (steepest-ascent) con 2-opt.
    Con reinicios>1 se convierte en Hill Climbing con reinicio aleatorio."""
    n = len(D)
    mejor_global, hist_global = None, []
    mejor_costo = float("inf")
    evals = 0
    for _ in range(reinicios):
        ruta = list(range(n))
        rng.shuffle(ruta)
        c = costo(ruta, D)
        hist = [c]
        for _ in range(max_iter):
            mejor_d, mi, mj = 0.0, None, None
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    d = delta_2opt(ruta, i, j, D)
                    evals += 1
                    if d < mejor_d - 1e-12:
                        mejor_d, mi, mj = d, i, j
            if mi is None:          # óptimo local
                break
            aplicar_2opt(ruta, mi, mj)
            c += mejor_d
            hist.append(c)
        if c < mejor_costo:
            mejor_costo, mejor_global = c, ruta[:]
        hist_global.extend(hist)
    return mejor_global, mejor_costo, hist_global, evals


def simulated_annealing_tsp(D, rng, T0=100.0, alpha=0.995, T_min=1e-3,
                            iter_por_T=50):
    """Recocido simulado con enfriamiento geométrico T <- alpha*T."""
    n = len(D)
    ruta = list(range(n))
    rng.shuffle(ruta)
    c = costo(ruta, D)
    mejor, mejor_c = ruta[:], c
    hist = [c]
    T = T0
    evals = 0
    while T > T_min:
        for _ in range(iter_por_T):
            i, j, d = vecino_aleatorio(ruta, D, rng)
            evals += 1
            if d < 0 or rng.random() < math.exp(-d / T):
                aplicar_2opt(ruta, i, j)
                c += d
                if c < mejor_c - 1e-12:
                    mejor, mejor_c = ruta[:], c
            hist.append(mejor_c)
        T *= alpha
    return mejor, mejor_c, hist, evals


# ---------------------------------------------------------------- Rastrigin
def rastrigin(x):
    return 10 * len(x) + sum(xi ** 2 - 10 * math.cos(2 * math.pi * xi) for xi in x)


def hc_rastrigin(rng, eta=0.1, iters=5000, dim=2):
    x = [rng.uniform(-5.12, 5.12) for _ in range(dim)]
    f = rastrigin(x)
    hist = [f]
    for _ in range(iters):
        y = [min(5.12, max(-5.12, xi + rng.gauss(0, eta))) for xi in x]
        fy = rastrigin(y)
        if fy < f:
            x, f = y, fy
        hist.append(f)
    return x, f, hist


def sa_rastrigin(rng, eta=0.5, T0=10.0, alpha=0.999, iters=5000, dim=2):
    x = [rng.uniform(-5.12, 5.12) for _ in range(dim)]
    f = rastrigin(x)
    mejor, mf = x[:], f
    hist = [mf]
    T = T0
    for _ in range(iters):
        y = [min(5.12, max(-5.12, xi + rng.gauss(0, eta))) for xi in x]
        fy = rastrigin(y)
        if fy < f or rng.random() < math.exp(-(fy - f) / max(T, 1e-12)):
            x, f = y, fy
            if f < mf:
                mejor, mf = x[:], f
        hist.append(mf)
        T *= alpha
    return mejor, mf, hist


# ---------------------------------------------------------------- Experimentos
def graficar_ruta(ax, ciudades, ruta, titulo, color):
    p = ciudades[ruta + [ruta[0]]]
    ax.plot(p[:, 0], p[:, 1], "-", color=color, lw=1.5, zorder=1)
    ax.scatter(ciudades[:, 0], ciudades[:, 1], c="white", edgecolors=GRIS, s=45, zorder=2)
    ax.scatter(*ciudades[ruta[0]], c="black", s=70, zorder=3)
    ax.set_title(titulo, fontsize=10)
    ax.set_aspect("equal")
    ax.tick_params(labelsize=7)


def main():
    ciudades = generar_ciudades(N_CIUDADES)
    D = matriz_distancias(ciudades)
    res = {"semilla": SEMILLA, "n_ciudades": N_CIUDADES, "corridas": N_CORRIDAS}

    # ---- 30 corridas por algoritmo
    datos = {"HC": [], "HC_reinicios": [], "SA": []}
    mejores = {}
    hist_rep = {}
    tiempos = {"HC": 0, "HC_reinicios": 0, "SA": 0}
    evals_tot = {"HC": 0, "HC_reinicios": 0, "SA": 0}
    for k in range(N_CORRIDAS):
        for nombre, fn in (
            ("HC", lambda r: hill_climbing_tsp(D, r)),
            ("HC_reinicios", lambda r: hill_climbing_tsp(D, r, reinicios=10)),
            ("SA", lambda r: simulated_annealing_tsp(D, r)),
        ):
            rng = random.Random(SEMILLA * 1000 + k)
            t0 = time.perf_counter()
            ruta, c, hist, ev = fn(rng)
            tiempos[nombre] += time.perf_counter() - t0
            evals_tot[nombre] += ev
            datos[nombre].append(c)
            if nombre not in mejores or c < mejores[nombre][1]:
                mejores[nombre] = (ruta, c)
                hist_rep[nombre] = hist

    resumen = {}
    for n, v in datos.items():
        v = np.array(v)
        resumen[n] = {
            "mejor": float(v.min()), "peor": float(v.max()),
            "media": float(v.mean()), "desv": float(v.std(ddof=1)),
            "mediana": float(np.median(v)),
            "tiempo_medio_s": tiempos[n] / N_CORRIDAS,
            "evals_medias": evals_tot[n] / N_CORRIDAS,
        }
    res["tsp"] = resumen
    res["tsp_mejores_rutas"] = {k: [int(x) for x in v[0]] for k, v in mejores.items()}

    # ---- Gráfica: rutas
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    etiquetas = {"HC": "Hill Climbing", "HC_reinicios": "HC + 10 reinicios",
                 "SA": "Simulated Annealing"}
    colores = {"HC": GRIS, "HC_reinicios": VERDE, "SA": AZUL}
    for ax, n in zip(axs, ["HC", "HC_reinicios", "SA"]):
        graficar_ruta(ax, ciudades, mejores[n][0],
                      f"{etiquetas[n]}\nmejor costo = {mejores[n][1]:.1f}", colores[n])
    plt.tight_layout()
    plt.savefig(f"{OUT}/rutas.png", dpi=160)
    plt.close()

    # ---- Gráfica: convergencia (mejor corrida)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(hist_rep["HC"], color=GRIS, label="Hill Climbing")
    ax[0].set_xlabel("Iteración (movimiento aceptado)")
    ax[0].set_ylabel("Costo de la ruta")
    ax[0].set_title("Convergencia - Hill Climbing")
    ax[0].grid(alpha=.3)
    ax[0].legend()
    ax[1].plot(hist_rep["SA"], color=AZUL, label="Simulated Annealing (mejor hallado)")
    ax[1].set_xlabel("Evaluación de vecino")
    ax[1].set_ylabel("Mejor costo")
    ax[1].set_title("Convergencia - Simulated Annealing")
    ax[1].grid(alpha=.3)
    ax[1].legend()
    plt.tight_layout()
    plt.savefig(f"{OUT}/convergencia.png", dpi=160)
    plt.close()

    # ---- Gráfica: boxplot comparación
    fig, ax = plt.subplots(figsize=(6.5, 4))
    bp = ax.boxplot([datos["HC"], datos["HC_reinicios"], datos["SA"]],
                    patch_artist=True, medianprops=dict(color="black", lw=2), tick_labels=["HC", "HC+reinicios", "SA"])
    for patch, col in zip(bp["boxes"], [GRIS, VERDE, AZUL]):
        patch.set_facecolor(col)
        patch.set_alpha(.6)
    ax.set_ylabel("Costo final (30 corridas)")
    ax.set_title("Comparación de resultados - TSP 25 ciudades")
    ax.grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/comparacion.png", dpi=160)
    plt.close()

    # ---- Sensibilidad de parámetros SA (T0, alpha), 10 corridas c/u
    T0s = [1, 10, 100]
    alphas = [0.90, 0.95, 0.99, 0.995]
    sens = {}
    for T0 in T0s:
        for a in alphas:
            vals = []
            for k in range(10):
                rng = random.Random(SEMILLA * 7 + k)
                _, c, _, _ = simulated_annealing_tsp(D, rng, T0=T0, alpha=a,
                                                     iter_por_T=30)
                vals.append(c)
            sens[f"T0={T0}, alpha={a}"] = {"media": float(np.mean(vals)),
                                            "mejor": float(np.min(vals))}
    res["sensibilidad_SA"] = sens
    M = np.array([[sens[f"T0={T0}, alpha={a}"]["media"] for a in alphas] for T0 in T0s])
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(M, cmap="Greys_r", aspect="auto")
    ax.set_xticks(range(len(alphas)), [str(a) for a in alphas])
    ax.set_yticks(range(len(T0s)), [str(t) for t in T0s])
    ax.set_xlabel("alpha (enfriamiento)")
    ax.set_ylabel("Temperatura inicial T0")
    ax.set_title("Costo medio de SA según parámetros")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                    color="white" if M[i, j] < M.mean() else "black", fontsize=9)
    plt.colorbar(im, ax=ax, label="costo medio")
    plt.tight_layout()
    plt.savefig(f"{OUT}/sensibilidad_sa.png", dpi=160)
    plt.close()

    # ---- Rastrigin (η)
    etas = [0.05, 0.1, 0.5, 1.0]
    ras = {}
    for eta in etas:
        hc_v = [hc_rastrigin(random.Random(SEMILLA + k), eta=eta)[1] for k in range(N_CORRIDAS)]
        sa_v = [sa_rastrigin(random.Random(SEMILLA + k), eta=eta)[1] for k in range(N_CORRIDAS)]
        ras[str(eta)] = {"HC_media": float(np.mean(hc_v)), "HC_mejor": float(np.min(hc_v)),
                         "SA_media": float(np.mean(sa_v)), "SA_mejor": float(np.min(sa_v))}
    res["rastrigin"] = ras
    _, _, h_hc = hc_rastrigin(random.Random(SEMILLA), eta=0.1)
    _, _, h_sa = sa_rastrigin(random.Random(SEMILLA), eta=0.5)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(h_hc, color=GRIS, label="HC (η=0.1)")
    ax[0].plot(h_sa, color=AZUL, label="SA (η=0.5)")
    ax[0].set_xlabel("Iteración")
    ax[0].set_ylabel("Mejor f(x)")
    ax[0].set_title("Rastrigin 2D - convergencia")
    ax[0].legend()
    ax[0].grid(alpha=.3)
    w = 0.35
    xs = np.arange(len(etas))
    ax[1].bar(xs - w / 2, [ras[str(e)]["HC_media"] for e in etas], w, color=GRIS, label="HC")
    ax[1].bar(xs + w / 2, [ras[str(e)]["SA_media"] for e in etas], w, color=AZUL, label="SA")
    ax[1].set_xticks(xs, [str(e) for e in etas])
    ax[1].set_xlabel("η (desviación del paso)")
    ax[1].set_ylabel("f media (30 corridas)")
    ax[1].set_title("Efecto de η")
    ax[1].legend()
    ax[1].grid(axis="y", alpha=.3)
    plt.tight_layout()
    plt.savefig(f"{OUT}/rastrigin.png", dpi=160)
    plt.close()

    with open(f"{OUT}/resultados.json", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, ensure_ascii=False)

    print(json.dumps({"tsp": resumen, "rastrigin": ras}, indent=2))


if __name__ == "__main__":
    main()
