"""Genera las instancias sinteticas (stochastic block model) de forma replicable.

Uso (desde la raiz del proyecto):
    python -m src.generate_instances [--salida data]

Categorias A-D con semillas 101, 202, 303. La semilla de cada instancia es
semilla*1000 + ord(categoria). Un nodo aislado se conecta a un nodo de su propio
bloque, porque los nodos sin aristas desaparecen al leer el archivo.
"""
import argparse
import random
from pathlib import Path

# categoria: (n, bloques, p_in, p_out)
CATEGORIES: dict[str, tuple[int, int, float, float]] = {
    "A": (15, 3, 0.50, 0.05),
    "B": (25, 4, 0.45, 0.04),
    "C": (25, 4, 0.40, 0.18),
    "D": (35, 5, 0.35, 0.08),
}
SEEDS = (101, 202, 303)


def block_of(n: int, groups: int) -> list[int]:
    base, extra = divmod(n, groups)
    block: list[int] = []
    for b in range(groups):
        block += [b] * (base + (1 if b < extra else 0))
    return block


def generate(category: str, seed: int) -> tuple[int, list[tuple[int, int]]]:
    n, groups, p_in, p_out = CATEGORIES[category]
    rng = random.Random(seed * 1000 + ord(category))
    block = block_of(n, groups)

    edges: set[tuple[int, int]] = set()
    for u in range(n):
        for v in range(u + 1, n):
            p = p_in if block[u] == block[v] else p_out
            if rng.random() < p:
                edges.add((u, v))

    degree = [0] * n
    for u, v in edges:
        degree[u] += 1
        degree[v] += 1
    for u in range(n):
        if degree[u] == 0:
            mates = [v for v in range(n) if v != u and block[v] == block[u]]
            v = rng.choice(mates)
            edges.add((min(u, v), max(u, v)))
            degree[u] += 1
            degree[v] += 1
    return n, sorted(edges)


def write(path: Path, edges: list[tuple[int, int]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for u, v in edges:
            f.write(f"{u:02d} {v:02d}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--salida", default="data", help="directorio de salida")
    out = Path(parser.parse_args().salida)
    out.mkdir(parents=True, exist_ok=True)
    for category in CATEGORIES:
        for seed in SEEDS:
            n, edges = generate(category, seed)
            write(out / f"{category}_s{seed}.txt", edges)
            print(f"{category}_s{seed}: n={n} m={len(edges)}")


if __name__ == "__main__":
    main()
