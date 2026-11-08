import collections
from dataclasses import dataclass
from pathlib import Path

type Edge = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Instance:
    name: str
    modules: tuple[str, ...]
    edges: tuple[Edge, ...]

    def __post_init__(self) -> None:
        if self.n < 2:
            raise ValueError(f"{self.name}: se necesitan al menos 2 modulos")
        if not self.edges:
            raise ValueError(f"{self.name}: la instancia no tiene dependencias")
        visited: set[tuple[int, int]] = set()
        for u, v, c in self.edges:
            if not 0 <= u < v < self.n:
                raise ValueError(f"{self.name}: arista ({u}, {v}) invalida, se exige 0 <= u < v < n")
            if c <= 0:
                raise ValueError(f"{self.name}: peso no positivo en ({u}, {v})")
            if (u, v) in visited:
                raise ValueError(f"{self.name}: par ({u}, {v}) repetido")
            visited.add((u, v))

    @classmethod
    def from_file(cls, path: str | Path) -> "Instance":
        path = Path(path)
        weight: collections.Counter[tuple[str, str]] = collections.Counter()
        with path.open(encoding="utf-8") as f:
            for linea in f:
                t = linea.split()
                if len(t) >= 2 and t[0] != t[1]:
                    w = int(t[2]) if len(t) > 2 else 1
                    weight[tuple(sorted(t[:2]))] += w
        names = sorted({v for par in weight for v in par})
        idx = {v: i for i, v in enumerate(names)}
        edges = sorted(
            (min(idx[a], idx[b]), max(idx[a], idx[b]), w) for (a, b), w in weight.items()
        )
        name = path.parent.name if path.name == "Calls.txt" else path.name
        return cls(name=name, modules=tuple(names), edges=tuple(edges))

    @property
    def n(self) -> int:
        return len(self.modules)

    @property
    def m(self) -> int:
        return len(self.edges)

    @property
    def density(self) -> float:
        return self.m / (self.n * (self.n - 1) / 2)

    def neighbors(self) -> list[list[int]]:
        nb: list[list[int]] = [[] for _ in range(self.n)]
        for u, v, _ in self.edges:
            nb[u].append(v)
            nb[v].append(u)
        return nb

    def components(self) -> list[list[int]]:
        nb = self.neighbors()
        visited = [False] * self.n
        comps: list[list[int]] = []
        for s in range(self.n):
            if visited[s]:
                continue
            visited[s] = True
            stack, comp = [s], []
            while stack:
                u = stack.pop()
                comp.append(u)
                for v in nb[u]:
                    if not visited[v]:
                        visited[v] = True
                        stack.append(v)
            comps.append(sorted(comp))
        return comps