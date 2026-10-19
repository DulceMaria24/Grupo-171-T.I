"""
Módulo de Simulación Interactiva para Problemas de Búsqueda en Grafos.

Este script implementa un dashboard interactivo utilizando NetworkX, Matplotlib 
y widgets (Sliders). Permite simular y controlar paso a paso el avance de 
dos agentes inteligentes en escenarios reales:
    1. Logística de Transporte Agrícola (Grafo Dirigido Ponderado).
    2. Sistema de Navegación Vehicular Autónomo en Malla Urbana (Grid 3x3).

Incluye además paneles dinámicos de explicación textual que detallan en tiempo 
real el estado actual (S), la acción ejecutada (A), el costo acumulado y la meta (G).
"""

from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import networkx as nx

# ==============================================================================
# 1. MODELADO Y CÁLCULO DE RUTAS ÓPTIMAS (ALGORITMO DE DIJKSTRA)
# ==============================================================================

# --- Problema 1: Logística Agrícola ---
grafo_agricola: nx.DiGraph = nx.DiGraph()
conexiones_agricola: List[Tuple[str, str, int]] = [
    ("Atlautla", "Ozumba", 10),
    ("Atlautla", "Juchitepec", 25),
    ("Ozumba", "Tepetlixpa", 8),
    ("Ozumba", "Amecameca", 18),
    ("Juchitepec", "Amecameca", 15),
    ("Juchitepec", "Tepetlixpa", 22),
    ("Tepetlixpa", "Cuautla", 30),
    ("Amecameca", "Cuautla", 40),
]

# Poblar el grafo agrícola con nodos y aristas ponderadas por tiempo (minutos)
for origen, destino, costo in conexiones_agricola:
    grafo_agricola.add_edge(origen, destino, weight=costo)

# Cálculo del camino óptimo utilizando el algoritmo de Dijkstra integrado en NetworkX
camino_g1: List[str] = nx.shortest_path(
    grafo_agricola, source="Atlautla", target="Cuautla", weight="weight"
)

# Coordenadas geográficas estáticas simuladas para la visualización del grafo 1
posiciones_g1: Dict[str, Tuple[int, int]] = {
    "Atlautla": (0, 1),
    "Juchitepec": (1, 2),
    "Ozumba": (1, 0),
    "Amecameca": (2, 2),
    "Tepetlixpa": (2, 0),
    "Cuautla": (3, 1),
}


# --- Problema 2: Navegación Urbana (Grid 3x3) ---
grafo_urbano: nx.DiGraph = nx.DiGraph()
costos_trafico_urbano: Dict[Tuple[Tuple[int, int], Tuple[int, int]], int] = {
    ((0, 0), (1, 0)): 5,
    ((0, 0), (0, 1)): 2,
    ((1, 0), (2, 0)): 8,
    ((1, 0), (1, 1)): 3,
    ((0, 1), (1, 1)): 1,
    ((0, 1), (0, 2)): 6,
    ((1, 1), (2, 1)): 4,
    ((1, 1), (1, 2)): 2,
    ((2, 0), (2, 1)): 3,
    ((2, 1), (2, 2)): 2,
    ((0, 2), (1, 2)): 7,
    ((1, 2), (2, 2)): 1,
}

# Construcción de la malla urbana de 3x3 nodos con restricciones de límites
for x in range(3):
    for y in range(3):
        for dx, dy in [(1, 0), (0, 1)]:  # Movimientos hacia la derecha y arriba
            vecino_pos = (x + dx, y + dy)
            if vecino_pos[0] < 3 and vecino_pos[1] < 3:
                costo_arista = costos_trafico_urbano.get(
                    ((x, y), vecino_pos), 5
                )
                grafo_urbano.add_edge((x, y), vecino_pos, weight=costo_arista)

# Cálculo de la ruta óptima para el vehículo autónomo
camino_g2: List[Tuple[int, int]] = nx.shortest_path(
    grafo_urbano, source=(0, 0), target=(2, 2), weight="weight"
)

# Posiciones basadas en las coordenadas cartesianas (x, y) de la cuadrícula
posiciones_g2: Dict[Tuple[int, int], Tuple[int, int]] = {
    node: (node[0], node[1]) for node in grafo_urbano.nodes()
}


# ==============================================================================
# 2. CONFIGURACIÓN DE LA INTERFAZ GRÁFICA (DASHBOARD MULTIPANEL)
# ==============================================================================

# Crear la figura principal dividida en dos subgráficos (uno para cada problema)
fig, (ax_grafico1, ax_grafico2) = plt.subplots(1, 2, figsize=(15, 7))

# Ajustar los márgenes inferiores para dar espacio a los sliders y cajas de texto
plt.subplots_adjust(bottom=0.32, wspace=0.3)

# ------------------------------------------------------------------------------
# Renderizado Estático del Grafo 1: Logística Agrícola
# ------------------------------------------------------------------------------
pesos_g1 = nx.get_edge_attributes(grafo_agricola, "weight")
nx.draw_networkx_nodes(
    grafo_agricola,
    posiciones_g1,
    node_color="skyblue",
    node_size=2000,
    ax=ax_grafico1,
)
nx.draw_networkx_labels(
    grafo_agricola, posiciones_g1, font_size=9, font_weight="bold", ax=ax_grafico1
)
nx.draw_networkx_edges(
    grafo_agricola,
    posiciones_g1,
    edgelist=grafo_agricola.edges(),
    arrowstyle="->",
    arrowsize=15,
    ax=ax_grafico1,
)
nx.draw_networkx_edge_labels(
    grafo_agricola,
    posiciones_g1,
    edge_labels=pesos_g1,
    font_size=8,
    ax=ax_grafico1,
)
ax_grafico1.set_title(
    "Simulador 1: Logística Agrícola", fontsize=11, fontweight="bold"
)

# ------------------------------------------------------------------------------
# Renderizado Estático del Grafo 2: Navegación Urbana
# ------------------------------------------------------------------------------
pesos_g2 = nx.get_edge_attributes(grafo_urbano, "weight")
nx.draw_networkx_nodes(
    grafo_urbano,
    posiciones_g2,
    node_color="lightgreen",
    node_size=1500,
    ax=ax_grafico2,
)
nx.draw_networkx_labels(
    grafo_urbano, posiciones_g2, font_size=8, font_weight="bold", ax=ax_grafico2
)
nx.draw_networkx_edges(
    grafo_urbano,
    posiciones_g2,
    edgelist=grafo_urbano.edges(),
    arrowstyle="->",
    arrowsize=15,
    ax=ax_grafico2,
)
nx.draw_networkx_edge_labels(
    grafo_urbano,
    posiciones_g2,
    edge_labels=pesos_g2,
    font_size=8,
    ax=ax_grafico2,
)
ax_grafico2.set_title(
    "Simulador 2: Navegación Urbana (Grid 3x3)", fontsize=11, fontweight="bold"
)

# Elementos dinámicos: marcadores que representan la posición actual de los agentes
(nodo_actual_1,) = ax_grafico1.plot([], [], "o", color="orange", markersize=25)
(nodo_actual_2,) = ax_grafico2.plot([], [], "o", color="coral", markersize=22)

# Cajas de texto flotantes para mostrar la explicación formal en tiempo real
caja_texto_exp_1 = fig.text(
    0.10,
    0.14,
    "",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.6),
)
caja_texto_exp_2 = fig.text(
    0.53,
    0.14,
    "",
    fontsize=9,
    bbox=dict(boxstyle="round", facecolor="lightcyan", alpha=0.6),
)


# ==============================================================================
# 3. CONTROLES INTERACTIVOS (SLIDERS) Y LÓGICA DE ACTUALIZACIÓN DINÁMICA
# ==============================================================================

# Definir las coordenadas espaciales de los sliders en la interfaz de la ventana
eje_slider1 = plt.axes([0.15, 0.05, 0.3, 0.03])
eje_slider2 = plt.axes([0.58, 0.05, 0.3, 0.03])

# Instanciar los sliders interactivos controlados por pasos discretos (valstep=1)
slider_control_g1 = Slider(
    eje_slider1,
    "Paso Sim. 1",
    0,
    len(camino_g1) - 1,
    valinit=0,
    valstep=1,
    color="orange",
)
slider_control_g2 = Slider(
    eje_slider2,
    "Paso Sim. 2",
    0,
    len(camino_g2) - 1,
    valinit=0,
    valstep=1,
    color="coral",
)


def actualizar_simulacion(val: float) -> None:
    """Función callback ejecutada automáticamente cada vez que el usuario

    manipula cualquiera de los dos sliders interactivos.

    Actualiza la posición gráfica del agente en el grafo correspondiente y
    genera un reporte analítico detallando el estado (S), la acción (A),
    el costo acumulado y la condición de meta (G).

    Args:
        val (float): Valor numérico actual del slider movido (ignorado ya que
          se obtienen directamente las propiedades de los objetos Slider).
    """
    # --- 1. Procesamiento de Estado para el Grafo 1 (Logística Agrícola) ---
    paso_actual_1 = int(slider_control_g1.val)
    nodo_actual_nombre_1 = camino_g1[paso_actual_1]
    coord_x1, coord_y1 = posiciones_g1[nodo_actual_nombre_1]
    nodo_actual_1.set_data([coord_x1], [coord_y1])

    costo_acumulado_1 = 0
    descripcion_accion_1 = "Vehículo listo en la parcela de producción inicial."
    if paso_actual_1 > 0:
        for idx in range(paso_actual_1):
            costo_acumulado_1 += grafo_agricola[camino_g1[idx]][
                camino_g1[idx + 1]
            ]["weight"]
        descripcion_accion_1 = f"Transitar desde '{camino_g1[paso_actual_1-1]}' hacia '{nodo_actual_nombre_1}'."

    reporte_explicativo_1 = (
        f"[PROBLEMA 1: LOGÍSTICA AGRÍCOLA]\n"
        f"• Estado Actual (S): {nodo_actual_nombre_1}\n"
        f"• Acción Ejecutada (A): {descripcion_accion_1}\n"
        f"• Costo Acumulado: {costo_acumulado_1} minutos de traslado\n"
        f"• Meta (G): {'Alcanzada (Centro de Acopio en Cuautla)' if nodo_actual_nombre_1 == 'Cuautla' else 'En tránsito hacia el centro de distribución...'}"
    )
    caja_texto_exp_1.set_text(reporte_explicativo_1)

    # --- 2. Procesamiento de Estado para el Grafo 2 (Navegación Urbana) ---
    paso_actual_2 = int(slider_control_g2.val)
    nodo_actual_coord_2 = camino_g2[paso_actual_2]
    coord_x2, coord_y2 = posiciones_g2[nodo_actual_coord_2]
    nodo_actual_2.set_data([coord_x2], [coord_y2])

    costo_acumulado_2 = 0
    descripcion_accion_2 = (
        "Vehículo autónomo encendido en intersección origen (0,0)."
    )
    if paso_actual_2 > 0:
        for idx in range(paso_actual_2):
            costo_acumulado_2 += grafo_urbano[camino_g2[idx]][camino_g2[idx + 1]][
                "weight"
            ]
        descripcion_accion_2 = f"Avanzar de intersección {camino_g2[paso_actual_2-1]} a {nodo_actual_coord_2}."

    reporte_explicativo_2 = (
        f"[PROBLEMA 2: NAVEGACIÓN URBANA]\n"
        f"• Estado Actual (S): Intersección {nodo_actual_coord_2}\n"
        f"• Acción Ejecutada (A): {descripcion_accion_2}\n"
        f"• Costo de Tráfico Acumulado: {costo_acumulado_2} unidades ponderadas\n"
        f"• Meta (G): {'Destino alcanzado en (2,2)' if nodo_actual_coord_2 == (2,2) else 'Navegando el entorno urbano...'}"
    )
    caja_texto_exp_2.set_text(reporte_explicativo_2)

    # Forzar el redibujado inmediato de la interfaz gráfica
    fig.canvas.draw_idle()


# Vincular los eventos de cambio de los sliders con la función de actualización
slider_control_g1.on_changed(actualizar_simulacion)
slider_control_g2.on_changed(actualizar_simulacion)

# Inicializar los paneles de explicación ejecutando el estado base (paso 0)
actualizar_simulacion(0)

# ==============================================================================
# 4. EJECUCIÓN DE LA APLICACIÓN
# ==============================================================================
if __name__ == "__main__":
    # Mostrar la ventana interactiva completa con gráficos y widgets
    plt.show()