"""
Sección 4 - Algoritmos de búsqueda ciega (BFS / DFS)
Asignatura: Fundamentos de Inteligencia Artificial - UPA
Problema: resolver un laberinto en cuadrícula (estado = celda, acción = mover
arriba/abajo/izquierda/derecha, costo = 1 por paso).

Uso:  python busqueda_ciega.py
Genera: resultados_s4/*.png, resultados_s4/metricas.json y resultados_s4/simulacion_bfs_dfs.gif

SIMULACIÓN: al ejecutar se abre una ventana con la animación paso a paso de BFS y DFS
(clic sobre la ventana = pausar / reanudar). Opciones:
  python busqueda_ciega.py --sin-ventana      (solo guarda el GIF, útil sin pantalla)
  python busqueda_ciega.py --velocidad 3      (más rápido; 0.5 = más lento)
"""
import argparse
import json
import os
import time
from collections import deque

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap
import networkx as nx
import numpy as np

SALIDA = "resultados_s4"
os.makedirs(SALIDA, exist_ok=True)

# Laberinto principal: 0 = libre, 1 = pared, S = inicio, G = meta
LABERINTO = [
    "S0001000000000",
    "0110101111110",
    "0100100000010",
    "0101111011010",
    "0100000010010",
    "0111111010110",
    "0000001010000",
    "1110101011110",
    "0000101000010",
    "0111101111010",
    "0000000000G10",
]
# Se normaliza a ancho constante (las filas tienen 13 o 14 caracteres)
ANCHO = max(len(f) for f in LABERINTO)
LABERINTO = [f.ljust(ANCHO, "1") for f in LABERINTO]

MOVS = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # arriba, derecha, abajo, izquierda


def cargar(laberinto):
    rejilla = np.zeros((len(laberinto), len(laberinto[0])), dtype=int)
    inicio = meta = None
    for r, fila in enumerate(laberinto):
        for c, ch in enumerate(fila):
            if ch == "1":
                rejilla[r, c] = 1
            elif ch == "S":
                inicio = (r, c)
            elif ch == "G":
                meta = (r, c)
    return rejilla, inicio, meta


def vecinos(rejilla, estado):
    """Generador de estados sucesores (acciones legales)."""
    r, c = estado
    for dr, dc in MOVS:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rejilla.shape[0] and 0 <= nc < rejilla.shape[1] and rejilla[nr, nc] == 0:
            yield (nr, nc)


def reconstruir(padres, meta):
    camino, actual = [], meta
    while actual is not None:
        camino.append(actual)
        actual = padres[actual]
    return camino[::-1]


def bfs(rejilla, inicio, meta, trace=False):
    t0 = time.perf_counter()
    frontera = deque([inicio])
    padres = {inicio: None}
    orden, hist = [], []
    max_frontera = 1
    while frontera:
        max_frontera = max(max_frontera, len(frontera))
        actual = frontera.popleft()
        orden.append(actual)
        if actual != meta:
            for v in vecinos(rejilla, actual):
                if v not in padres:
                    padres[v] = actual
                    frontera.append(v)
        if trace:
            hist.append(list(frontera))   # foto de la frontera tras cada expansión
        if actual == meta:
            break
    dt = time.perf_counter() - t0
    camino = reconstruir(padres, meta) if meta in padres else []
    return dict(nombre="BFS", camino=camino, expandidos=len(orden), orden=orden, hist=hist,
                tiempo_ms=dt * 1000, max_frontera=max_frontera, costo=len(camino) - 1)


def dfs(rejilla, inicio, meta, trace=False):
    t0 = time.perf_counter()
    frontera = [(inicio, None)]  # pila de (estado, padre)
    padres = {}
    orden, hist = [], []
    max_frontera = 1
    while frontera:
        max_frontera = max(max_frontera, len(frontera))
        actual, padre = frontera.pop()
        if actual in padres:
            continue
        padres[actual] = padre
        orden.append(actual)
        if actual != meta:
            # invertido para que el primer movimiento sea el que se explora primero
            for v in reversed(list(vecinos(rejilla, actual))):
                if v not in padres:
                    frontera.append((v, actual))
        if trace:
            hist.append(list({e for e, _ in frontera if e not in padres}))
        if actual == meta:
            break
    dt = time.perf_counter() - t0
    camino = reconstruir(padres, meta) if meta in padres else []
    return dict(nombre="DFS", camino=camino, expandidos=len(orden), orden=orden, hist=hist,
                tiempo_ms=dt * 1000, max_frontera=max_frontera, costo=len(camino) - 1)


def a_grafo(rejilla):
    """Representación explícita del espacio de estados como grafo NetworkX."""
    G = nx.Graph()
    for r in range(rejilla.shape[0]):
        for c in range(rejilla.shape[1]):
            if rejilla[r, c] == 0:
                G.add_node((r, c))
                for v in vecinos(rejilla, (r, c)):
                    G.add_edge((r, c), v)
    return G


def medir(func, rejilla, inicio, meta, repeticiones=200):
    """Promedia el tiempo de ejecución para reducir ruido."""
    res = func(rejilla, inicio, meta)
    ts = []
    for _ in range(repeticiones):
        ts.append(func(rejilla, inicio, meta)["tiempo_ms"])
    res["tiempo_ms"] = float(np.mean(ts))
    return res


# ----------------------------- Gráficas -----------------------------------
AZUL, VERDE, GRIS = "#1F4E9E", "#2E9E5B", "#D9DEE7"


def dibujar_laberinto(ax, rejilla, res, inicio, meta, titulo):
    cmap = ListedColormap(["#FFFFFF", "#2B2F3A"])
    ax.imshow(rejilla, cmap=cmap)
    # celdas expandidas (exploración)
    for (r, c) in res["orden"]:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color=GRIS, zorder=1))
    # camino solución
    if res["camino"]:
        ys, xs = zip(*res["camino"])
        ax.plot(xs, ys, color=AZUL, lw=3, zorder=3)
    ax.scatter(inicio[1], inicio[0], c=VERDE, s=140, zorder=4, label="Inicio")
    ax.scatter(meta[1], meta[0], c="#D1342F", s=140, marker="*", zorder=4, label="Meta")
    ax.set_title(f"{titulo}\nexpandidos={res['expandidos']}  |  longitud={res['costo']}", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def figuras(rejilla, inicio, meta, rb, rd, G):
    fig, axs = plt.subplots(1, 2, figsize=(11, 5))
    dibujar_laberinto(axs[0], rejilla, rb, inicio, meta, "BFS (cola FIFO)")
    dibujar_laberinto(axs[1], rejilla, rd, inicio, meta, "DFS (pila LIFO)")
    axs[0].legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/fig1_laberinto_bfs_dfs.png", dpi=170)
    plt.close(fig)

    # Grafo del espacio de estados con camino de BFS resaltado
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pos = {n: (n[1], -n[0]) for n in G.nodes}
    nx.draw_networkx_edges(G, pos, ax=ax, edge_color="#B8C0CF", width=1)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=28, node_color="#8FA6CF")
    cam = rb["camino"]
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=list(zip(cam, cam[1:])), edge_color=AZUL, width=3)
    nx.draw_networkx_nodes(G, pos, nodelist=[inicio], node_color=VERDE, node_size=120, ax=ax)
    nx.draw_networkx_nodes(G, pos, nodelist=[meta], node_color="#D1342F", node_size=120, node_shape="*", ax=ax)
    ax.set_title(f"Grafo del espacio de estados (|V|={G.number_of_nodes()}, |E|={G.number_of_edges()})")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/fig2_grafo_estados.png", dpi=170)
    plt.close(fig)

    # Comparación de métricas
    etiquetas = ["Nodos expandidos", "Longitud del camino", "Frontera máx.", "Tiempo (ms ×100)"]
    vb = [rb["expandidos"], rb["costo"], rb["max_frontera"], rb["tiempo_ms"] * 100]
    vd = [rd["expandidos"], rd["costo"], rd["max_frontera"], rd["tiempo_ms"] * 100]
    x = np.arange(len(etiquetas)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    b1 = ax.bar(x - w / 2, vb, w, label="BFS", color=AZUL)
    b2 = ax.bar(x + w / 2, vd, w, label="DFS", color=VERDE)
    ax.bar_label(b1, fmt="%.0f", fontsize=8); ax.bar_label(b2, fmt="%.0f", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_title("Comparación BFS vs DFS en el laberinto principal")
    ax.legend(); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/fig3_comparacion.png", dpi=170)
    plt.close(fig)


# ----------------------- Experimento de escalabilidad -----------------------
def laberinto_aleatorio(n, densidad, semilla):
    rng = np.random.default_rng(semilla)
    while True:
        rej = (rng.random((n, n)) < densidad).astype(int)
        rej[0, 0] = 0; rej[n - 1, n - 1] = 0
        ini, met = (0, 0), (n - 1, n - 1)
        if bfs(rej, ini, met)["camino"]:
            return rej, ini, met


def escalabilidad():
    tamanos = [10, 20, 30, 40, 50, 60]
    filas = []
    for n in tamanos:
        eb, ed, cb, cd = [], [], [], []
        for s in range(8):
            rej, ini, met = laberinto_aleatorio(n, 0.25, 100 * n + s)
            rb, rd = bfs(rej, ini, met), dfs(rej, ini, met)
            eb.append(rb["expandidos"]); ed.append(rd["expandidos"])
            cb.append(rb["costo"]); cd.append(rd["costo"])
        filas.append(dict(n=n, bfs_exp=float(np.mean(eb)), dfs_exp=float(np.mean(ed)),
                          bfs_costo=float(np.mean(cb)), dfs_costo=float(np.mean(cd))))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.2))
    ns = [f["n"] for f in filas]
    axs[0].plot(ns, [f["bfs_exp"] for f in filas], "o-", color=AZUL, label="BFS")
    axs[0].plot(ns, [f["dfs_exp"] for f in filas], "s-", color=VERDE, label="DFS")
    axs[0].set_title("Nodos expandidos vs tamaño (n×n)"); axs[0].set_xlabel("n"); axs[0].legend()
    axs[1].plot(ns, [f["bfs_costo"] for f in filas], "o-", color=AZUL, label="BFS")
    axs[1].plot(ns, [f["dfs_costo"] for f in filas], "s-", color=VERDE, label="DFS")
    axs[1].set_title("Longitud del camino vs tamaño (n×n)"); axs[1].set_xlabel("n"); axs[1].legend()
    for a in axs:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(f"{SALIDA}/fig4_escalabilidad.png", dpi=170)
    plt.close(fig)
    return filas


# ----------------------------- Simulación ---------------------------------
def _hex(c):
    c = c.lstrip("#")
    return [int(c[i:i + 2], 16) for i in (0, 2, 4)]


C_LIBRE, C_PARED = _hex("#FFFFFF"), _hex("#2B2F3A")
C_VISITADO, C_FRONTERA = _hex("#BFD0EE"), _hex("#F2B84B")
C_ACTUAL, C_CAMINO = _hex("#D1342F"), _hex("#1F4E9E")


def simular(rejilla, inicio, meta, velocidad=1.0, mostrar=True, archivo_gif=None):
    """Animación paso a paso: BFS (izq.) y DFS (der.) resolviendo el mismo laberinto.
    Amarillo = frontera (por explorar) · azul claro = visitado · rojo = nodo que se expande ·
    azul fuerte = camino solución."""
    rb = bfs(rejilla, inicio, meta, trace=True)
    rd = dfs(rejilla, inicio, meta, trace=True)
    res = [rb, rd]
    pasos = max(len(rb["orden"]), len(rd["orden"]))
    largo_camino = max(len(rb["camino"]), len(rd["camino"]))
    espera = 12
    total = pasos + largo_camino + espera

    fig, axs = plt.subplots(1, 2, figsize=(11, 5.8))
    fig.suptitle("Simulación de búsqueda ciega en un laberinto", fontsize=13, fontweight="bold", color=AZUL)
    ims, lineas, textos = [], [], []
    for ax, r in zip(axs, res):
        im = ax.imshow(np.zeros((*rejilla.shape, 3), dtype=np.uint8), interpolation="nearest")
        ln, = ax.plot([], [], color=AZUL, lw=3.2, zorder=3)
        ax.scatter(inicio[1], inicio[0], c=VERDE, s=150, zorder=4)
        ax.scatter(meta[1], meta[0], c="#D1342F", s=170, marker="*", zorder=4)
        ax.set_xticks([]); ax.set_yticks([])
        textos.append(ax.set_title("", fontsize=10))
        ims.append(im); lineas.append(ln)
    leyenda = [plt.Rectangle((0, 0), 1, 1, color=np.array(c) / 255)
               for c in (C_FRONTERA, C_VISITADO, C_ACTUAL, C_CAMINO)]
    fig.legend(leyenda, ["Frontera", "Visitado", "Expandiendo", "Camino"], loc="lower center", ncol=4,
               fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))

    def imagen(r, k):
        img = np.empty((*rejilla.shape, 3), dtype=np.uint8)
        img[:] = C_LIBRE
        img[rejilla == 1] = C_PARED
        for (a, b) in r["orden"][:k]:
            img[a, b] = C_VISITADO
        if 0 < k <= len(r["hist"]):
            for (a, b) in r["hist"][k - 1]:
                img[a, b] = C_FRONTERA
        if 0 < k <= len(r["orden"]):
            a, b = r["orden"][k - 1]
            img[a, b] = C_ACTUAL
        return img

    def cuadro(f):
        for i, r in enumerate(res):
            n = len(r["orden"])
            k = min(f, n)
            ims[i].set_data(imagen(r, k))
            fase = "buscando" if f < n else "meta encontrada"
            front = len(r["hist"][k - 1]) if 0 < k <= len(r["hist"]) else 1
            if f >= pasos:
                p = min(f - pasos + 1, len(r["camino"]))
                ys, xs = zip(*r["camino"][:p])
                lineas[i].set_data(xs, ys)
            else:
                lineas[i].set_data([], [])
            textos[i].set_text(f"{r['nombre']} · {fase}\nexpandidos={k}  frontera={front}  "
                               f"longitud camino={r['costo'] if f >= n else '?'}")
        return ims + lineas + textos

    anim = FuncAnimation(fig, cuadro, frames=total, interval=140 / velocidad, blit=False, repeat=True)

    estado = {"pausa": False}

    def al_clic(_):
        if estado["pausa"]:
            anim.resume()
        else:
            anim.pause()
        estado["pausa"] = not estado["pausa"]

    fig.canvas.mpl_connect("button_press_event", al_clic)

    if archivo_gif:
        anim.save(archivo_gif, writer=PillowWriter(fps=max(1, round(7 * velocidad))), dpi=90)
    # fotogramas fijos para el reporte (inicio, mitad, final de la búsqueda, camino)
    marcas = [3, pasos // 3, (2 * pasos) // 3, total - 1]
    fig2, axs2 = plt.subplots(2, 2, figsize=(11, 7.4))
    for ax2, f in zip(axs2.ravel(), marcas):
        cuadro(f)
        fig.canvas.draw()
        foto = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        no_blanco = np.where(foto.min(axis=2) < 250)
        foto = foto[no_blanco[0].min():no_blanco[0].max() + 1, no_blanco[1].min():no_blanco[1].max() + 1]
        ax2.imshow(foto)
        ax2.axis("off")
    fig2.tight_layout()
    fig2.savefig(f"{SALIDA}/fig5_fotogramas_simulacion.png", dpi=150)
    plt.close(fig2)
    if mostrar and plt.get_backend().lower() not in ("agg", "pdf", "svg", "cairo", "ps", "template"):
        plt.show()
    plt.close(fig)
    return anim


def main():
    ap = argparse.ArgumentParser(description="BFS vs DFS con simulación animada")
    ap.add_argument("--sin-ventana", action="store_true", help="no abrir la ventana; solo guardar el GIF")
    ap.add_argument("--velocidad", type=float, default=1.0, help="velocidad de la animación (1 = normal)")
    args = ap.parse_args()
    rejilla, inicio, meta = cargar(LABERINTO)
    G = a_grafo(rejilla)
    rb = medir(bfs, rejilla, inicio, meta)
    rd = medir(dfs, rejilla, inicio, meta)
    figuras(rejilla, inicio, meta, rb, rd, G)
    esc = escalabilidad()
    simular(rejilla, inicio, meta, velocidad=args.velocidad, mostrar=not args.sin_ventana,
            archivo_gif=f"{SALIDA}/simulacion_bfs_dfs.gif")

    resumen = {
        "laberinto": {"filas": rejilla.shape[0], "columnas": rejilla.shape[1],
                      "nodos_libres": G.number_of_nodes(), "aristas": G.number_of_edges(),
                      "inicio": inicio, "meta": meta},
        "BFS": {k: rb[k] for k in ("expandidos", "costo", "max_frontera", "tiempo_ms")},
        "DFS": {k: rd[k] for k in ("expandidos", "costo", "max_frontera", "tiempo_ms")},
        "escalabilidad": esc,
    }
    with open(f"{SALIDA}/metricas.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"{'Algoritmo':<10}{'Expandidos':>12}{'Longitud':>10}{'Frontera':>10}{'ms':>9}")
    for r in (rb, rd):
        print(f"{r['nombre']:<10}{r['expandidos']:>12}{r['costo']:>10}{r['max_frontera']:>10}{r['tiempo_ms']:>9.3f}")


if __name__ == "__main__":
    main()
