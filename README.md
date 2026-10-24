# Sección 3 - Búsqueda Local (Hill Climbing y Simulated Annealing)

Fundamentos de Inteligencia Artificial · Universidad Politécnica de Atlautla · Grupo 171
Alumna: Dulce María Sánchez Fragozo

## Contenido
- `busqueda_local.py`: TSP de 25 ciudades (vecindad 2-opt) y función Rastrigin 2D, con HC, HC con reinicios y SA.
- `simulacion.py`: animación de HC y SA sobre el TSP (GIF y cuadros fijos).
- `resultados/`: gráficas (PNG) y `resultados.json` con las métricas de 30 corridas.
- `FIA_G171_Sanchez_S3.pdf`: informe de la actividad.

## Cómo ejecutar
```
pip install -r requirements.txt
python busqueda_local.py
```
Las semillas son fijas (171), por lo que los resultados son reproducibles. Tarda unos 25 segundos.

## Simulación animada
```
python simulacion.py
```
Genera `resultados/simulacion_tsp.gif` (HC y SA lado a lado) y `resultados/simulacion_cuadros.png`. Tarda unos 30 segundos.
