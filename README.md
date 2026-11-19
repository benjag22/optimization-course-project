# Software Clustering Problem con MILP

Formulaciones MILP del Software Clustering Problem (Köhler et al., 2013) resueltas con Gurobi y CPLEX.

## Estructura

- `src/instance.py`, `src/preprocess.py`, `src/mq.py`. Lectura del grafo, reduccion de hojas y calculo de Turbo MQ.
- `src/models/MILP.py`. Opciones y listas de variables y cortes, comunes a ambos solvers.
- `src/models/MILP_gurobi.py`, `src/models/MILP_cplex.py`. Mismo modelo escrito con gurobipy y docplex.
- `src/generate_instances.py`. Instancias sinteticas A-D (stochastic block model, semillas 101, 202, 303).
- `src/run_benchmark.py`. Experimentos 0 a 4, agrega filas a `results/results.csv`.
- `data/`. Instancias `A_s101.txt` ... `D_s303.txt`, `example6.txt` y `cia++.txt`.

## Uso

Todos los comandos se ejecutan desde la raiz del proyecto.

```
uv sync --no-install-project
python -m src.generate_instances
python -m src.run_benchmark --experimento 0 --solver ambos
python -m src.run_benchmark --experimento 1 2 3 4 --solver ambos --reanudar
```

Opciones del runner

- `--experimento` uno o mas valores entre 0 y 4. El 0 es la validacion.
- `--solver` gurobi, cplex o ambos.
- `--instancias` filtra por nombre sin extension, por ejemplo `A_s101 cia++`.
- `--reanudar` omite las corridas que ya estan en el CSV.
- `--hilos` (4 por defecto) y `--limite` en segundos (3600 por defecto).

## Reglas de comparacion

- MIPGap relativo y absoluto de 1e-6 en ambos solvers.
- El tiempo es de reloj de pared y mide solo la resolucion.
- Toda solucion se recalcula con `mq()` y se avisa si difiere del objetivo en mas de 1e-6.
- Con `exact_K`, `min_size` o `max_size` se apagan `preprocess`, `cuts_one_cluster` y `cuts_two_clusters`, porque los teoremas 4.1 y 4.2 dejan de valer.
- Experimento 1 corre `cia++` con `cuts_two_clusters=False` por el tamaño de las desigualdades (47).

## Validacion

Optimos esperados para el experimento 0. `compiler` 1.5065, `lab4` 3.4000 y `example6` 12/7 (1.7143). Las instancias `compiler` y `lab4` vienen del repositorio SoftwareMaintenanceLab. Si su archivo no esta en `data/` como `compiler.txt` o `lab4.txt`, la corrida se omite.

## CPLEX

`docplex` es Python puro y usa la instalacion de CPLEX Studio a traves de las variables `CPLEX_STUDIO_DIR*`. El paquete `cplex` de PyPI no se incluye porque es la edicion Community con limite de tamaño.
