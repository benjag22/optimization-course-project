"""Preprocesamiento de hojas (Köhler et al., 2013, secc. 4.1)

Una hoja u (grado 1, sin auto-lazo) siempre queda en el cluster de su vecino v y
se elimina u y su arista pasa a ser un auto-lazo de v con el mismo peso de forma que
el MQ del grafo reducido es igual al MQ del original una vez reinsertadas las hojas
"""
from dataclasses import dataclass
from src.instance import Instance

@dataclass(frozen=True, slots=True)
class Reduction:
    original: Instance
    reduced: Instance                   
    ties: tuple[int, ...]               
    a_original: tuple[int, ...]
    sheets: tuple[tuple[int, int], ...]   # (sheet, neighbor) by original indexes

    def expand(self, reduced_assignation: list[int]) -> list[int]:
        asig = [-1] * self.original.n
        for i, k in enumerate(reduced_assignation):
            asig[self.a_original[i]] = k
        for hoja, vecino in self.sheets:
            asig[hoja] = asig[vecino]
        return asig


def without_reducing(inst: Instance) -> Reduction:
    return Reduction(inst, inst, (0,) * inst.n, tuple(range(inst.n)), ())


def reduce_sheets(inst: Instance) -> Reduction:
    neighbors = [dict() for _ in range(inst.n)]  # vecinos[u][v] = c_uv
    for u, v, c in inst.edges:
        neighbors[u][v] = c
        neighbors[v][u] = c
    lazo = [0] * inst.n
    eliminated = [False] * inst.n
    sheets: list[tuple[int, int]] = []

    # Una sola pasada basta, pues un nodo que queda con grado 1 después de perder
    # una hoja ya tiene auto-lazo, y el teorema solo vale para nodos sin lazo.
    for u in range(inst.n):
        if lazo[u] == 0 and len(neighbors[u]) == 1:
            (v, c), = neighbors[u].items()
            del neighbors[v][u]
            neighbors[u].clear()
            lazo[v] += c
            eliminated[u] = True
            sheets.append((u, v))

    remain = [u for u in range(inst.n) if not eliminated[u]]
    new = {u: i for i, u in enumerate(remain)}
    edges = tuple(sorted(
        (new[u], new[v], c)
        for u in remain for v, c in neighbors[u].items() if u < v
    ))
    if not edges:
        return without_reducing(inst)
    reduced = Instance(
        name=f"{inst.name}_red",
        modules=tuple(inst.modules[u] for u in remain),
        edges=edges,
    )
    return Reduction(inst, reduced, tuple(lazo[u] for u in remain), tuple(remain), tuple(sheets))