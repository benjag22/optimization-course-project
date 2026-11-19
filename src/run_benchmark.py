"""Benchmark Gurobi vs CPLEX para el SCP.

Uso (desde la raiz del proyecto):
    python -m src.run_benchmark --experimento 0 --solver ambos
    python -m src.run_benchmark --experimento 1 2 3 4 --solver ambos --reanudar
    python -m src.run_benchmark --experimento 1 --solver gurobi --instancias A_s101 B_s202

Experimentos
    0  validacion (optimos conocidos)
    1  benchmark base, Options() en A-D y cia++ sin cortes (47)
    2  ablacion, 4 variantes en B y C
    3  K-SCP, MAX_K 3/4/5 con exact_K en C y cia++
    4  tamaño acotado (L,U) = (4,8) y (5,7) en B y C

Cada corrida agrega una fila a results/results.csv.
"""
import argparse
import csv
import dataclasses
import math
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.instance import Instance
from src.models.MILP import MILPData, Options
from src.mq import mq

GAP = 1e-6  # MIPGap relativo y absoluto, igual en ambos solvers
SEEDS = (101, 202, 303)
VALIDATION = {"compiler": 1.5065, "lab4": 3.4000, "example6": 12 / 7}
VALIDATION_TOL = 1e-4  # los optimos de compiler se conocen con 4 decimales
MQ_TOL = 1e-6

COLUMNS = [
    "experimento", "instancia", "categoria", "modelo", "config", "solver",
    "n_orig", "m_orig", "n_red", "m_red",
    "tiempo_seg", "estado", "best_obj", "best_bound", "mip_gap_pct",
    "mq_verificado",
    "num_clusters_activos", "tam_min_cluster", "tam_max_cluster",
]


@dataclass(frozen=True, slots=True)
class Run:
    experimento: int
    nombre: str
    config: str
    options: Options


@dataclass(frozen=True, slots=True)
class Outcome:
    tiempo: float
    estado: str
    obj: float | None
    bound: float | None
    gap: float | None  # fraccion
    asignacion: list[int] | None


def finite(x: float | None) -> float | None:
    return x if x is not None and math.isfinite(x) else None


def safe_options(o: Options) -> Options:
    """Con exact_K, min_size o max_size los teoremas 4.1 y 4.2 dejan de valer."""
    if o.exact_K or o.min_size is not None or o.max_size is not None:
        return dataclasses.replace(
            o, preprocess=False, cuts_one_cluster=False, cuts_two_clusters=False
        )
    return o


def synthetic(categories: str) -> list[str]:
    return [f"{c}_s{s}" for c in categories for s in SEEDS]


def category(nombre: str) -> str:
    if nombre in VALIDATION:
        return "validacion"
    m = re.fullmatch(r"([A-D])_s\d+", nombre)
    return m.group(1) if m else "estres"


def plan(exp: int) -> list[Run]:
    if exp == 0:
        return [Run(0, n, "default", Options()) for n in VALIDATION]
    if exp == 1:
        runs = [Run(1, n, "default", Options()) for n in synthetic("ABCD")]
        runs.append(Run(1, "cia++", "sin_cortes_2", Options(cuts_two_clusters=False)))
        return runs
    if exp == 2:
        variants = {
            "puro": Options(preprocess=False, symmetry=False,
                            cuts_one_cluster=False, cuts_two_clusters=False),
            "hojas": Options(preprocess=True, symmetry=False,
                             cuts_one_cluster=False, cuts_two_clusters=False),
            "hojas_simetria": Options(preprocess=True, symmetry=True,
                                      cuts_one_cluster=False, cuts_two_clusters=False),
            "completo": Options(),
        }
        return [Run(2, n, cfg, o) for n in synthetic("BC") for cfg, o in variants.items()]
    if exp == 3:
        return [
            Run(3, n, f"K{k}_exacto", safe_options(Options(MAX_K=k, exact_K=True)))
            for k in (3, 4, 5)
            for n in synthetic("C") + ["cia++"]
        ]
    if exp == 4:
        return [
            Run(4, n, f"L{lo}_U{hi}", safe_options(Options(min_size=lo, max_size=hi)))
            for lo, hi in ((4, 8), (5, 7))
            for n in synthetic("BC")
        ]
    raise ValueError(f"experimento invalido {exp}")


def model_name(o: Options) -> str:
    if o.min_size is not None or o.max_size is not None:
        return "C-MILP2"
    if o.exact_K and o.MAX_K is not None:
        return "MILP3"
    return "MILP2"


def solve_gurobi(data: MILPData, threads: int, limit: float) -> Outcome:
    from gurobipy import GRB
    from src.models.MILP_gurobi import MILPGurobi

    modelo = MILPGurobi(data)
    g = modelo.model
    g.Params.OutputFlag = 0
    g.Params.Threads = threads
    g.Params.TimeLimit = limit
    g.Params.MIPGap = GAP
    g.Params.MIPGapAbs = GAP

    t0 = time.perf_counter()
    g.optimize()
    elapsed = time.perf_counter() - t0

    if g.Status == GRB.OPTIMAL:
        estado = "OPTIMAL"
    elif g.Status == GRB.TIME_LIMIT:
        estado = "TIME_LIMIT"
    elif g.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        estado = "INFEASIBLE"
    else:
        estado = f"OTRO_{g.Status}"

    has_solution = g.SolCount > 0
    obj = g.ObjVal if has_solution else None
    gap = g.MIPGap if has_solution else None
    try:
        bound = g.ObjBound
    except Exception:
        bound = None
    asignacion = modelo.assignation() if has_solution else None
    g.dispose()
    return Outcome(elapsed, estado, finite(obj), finite(bound), finite(gap), asignacion)


def solve_cplex(data: MILPData, threads: int, limit: float) -> Outcome:
    from src.models.MILP_cplex import MILPCplex

    modelo = MILPCplex(data)
    c = modelo.model
    c.parameters.threads = threads
    c.parameters.timelimit = limit
    c.parameters.mip.tolerances.mipgap = GAP
    c.parameters.mip.tolerances.absmipgap = GAP

    t0 = time.perf_counter()
    sol = c.solve(log_output=False)
    elapsed = time.perf_counter() - t0

    sd = c.solve_details
    code = sd.status_code
    if code in (101, 102):
        estado = "OPTIMAL"
    elif code in (107, 108):
        estado = "TIME_LIMIT"
    elif code == 103:
        estado = "INFEASIBLE"
    else:
        estado = f"OTRO_{code}"

    obj = c.objective_value if sol is not None else None
    gap = sd.mip_relative_gap if sol is not None else None
    asignacion = modelo.assignation() if sol is not None else None
    c.end()
    return Outcome(elapsed, estado, finite(obj), finite(sd.best_bound), finite(gap), asignacion)


SOLVERS: dict[str, Callable[[MILPData, int, float], Outcome]] = {
    "gurobi": solve_gurobi,
    "cplex": solve_cplex,
}


def execute(run: Run, inst: Instance, solver: str, threads: int, limit: float) -> dict[str, object]:
    data = MILPData.from_instance(inst, run.options)
    out = SOLVERS[solver](data, threads, limit)

    row: dict[str, object] = {
        "experimento": run.experimento,
        "instancia": run.nombre,
        "categoria": category(run.nombre),
        "modelo": model_name(run.options),
        "config": run.config,
        "solver": solver,
        "n_orig": inst.n,
        "m_orig": inst.m,
        "n_red": data.instance.n,
        "m_red": data.instance.m,
        "tiempo_seg": f"{out.tiempo:.3f}",
        "estado": out.estado,
        "best_obj": "" if out.obj is None else f"{out.obj:.6f}",
        "best_bound": "" if out.bound is None else f"{out.bound:.6f}",
        "mip_gap_pct": "" if out.gap is None else f"{100 * out.gap:.6f}",
        "mq_verificado": "",
        "num_clusters_activos": "",
        "tam_min_cluster": "",
        "tam_max_cluster": "",
    }

    if out.asignacion is not None:
        verified = mq(inst, out.asignacion)
        sizes = Counter(out.asignacion).values()
        row["mq_verificado"] = f"{verified:.6f}"
        row["num_clusters_activos"] = len(sizes)
        row["tam_min_cluster"] = min(sizes)
        row["tam_max_cluster"] = max(sizes)
        if out.obj is not None and abs(out.obj - verified) > MQ_TOL:
            print(
                f"AVISO {run.nombre} {solver} {run.config}: obj={out.obj:.8f} "
                f"mq()={verified:.8f}",
                file=sys.stderr,
            )

    if run.experimento == 0:
        expected = VALIDATION[run.nombre]
        ok = out.estado == "OPTIMAL" and out.obj is not None and abs(out.obj - expected) <= VALIDATION_TOL
        print(f"VALIDACION {run.nombre} {solver} esperado={expected:.4f} {'OK' if ok else 'ERROR'}")
    return row


def done_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8", newline="") as f:
        return {
            (r["experimento"], r["instancia"], r["config"], r["solver"])
            for r in csv.DictReader(f)
        }


def append(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)


def fmt(seg: float) -> str:
    h, r = divmod(int(seg), 3600)
    m, s = divmod(r, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--experimento", type=int, nargs="+", choices=range(5), required=True)
    parser.add_argument("--solver", choices=("gurobi", "cplex", "ambos"), default="ambos")
    parser.add_argument("--instancias", nargs="+", help="nombres sin extension, ej. A_s101")
    parser.add_argument("--reanudar", action="store_true", help="omite corridas ya registradas")
    parser.add_argument("--hilos", type=int, default=4)
    parser.add_argument("--limite", type=float, default=3600.0, help="segundos por corrida")
    parser.add_argument("--data", default="data")
    parser.add_argument("--salida", default="results/results.csv")
    args = parser.parse_args(argv)

    data_dir, out_path = Path(args.data), Path(args.salida)
    solvers = ["gurobi", "cplex"] if args.solver == "ambos" else [args.solver]
    done = done_keys(out_path) if args.reanudar else set()

    pending: list[tuple[int, Run, Path, str]] = []
    for exp in args.experimento:
        for run in plan(exp):
            if args.instancias and run.nombre not in args.instancias:
                continue
            path = data_dir / f"{run.nombre}.txt"
            if not path.exists():
                print(f"OMITIDA {run.nombre}: no existe {path}", file=sys.stderr)
                continue
            for solver in solvers:
                if (str(exp), run.nombre, run.config, solver) in done:
                    continue
                pending.append((exp, run, path, solver))

    total = len(pending)
    t_start = time.perf_counter()
    for i, (exp, run, path, solver) in enumerate(pending, 1):
        inst = Instance.from_file(path)
        row = execute(run, inst, solver, args.hilos, args.limite)
        append(out_path, row)
        left = total - i
        avg = (time.perf_counter() - t_start) / i
        print(
            f"[{i}/{total}] exp{exp} {run.nombre} {run.config} {solver} {row['estado']} "
            f"obj={row['best_obj']} gap%={row['mip_gap_pct']} t={row['tiempo_seg']}s "
            f"| ETA~{fmt(avg * left)} max {fmt(left * args.limite)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
