from dataclasses import dataclass
from src.instance import Instance
from src.preprocess import Reduction, reduce_sheets, without_reducing


def everyone_can(u: int, w: int, symmetry: bool, neighbors: list[list[int]]) -> bool:
    return not symmetry or w <= min(neighbors[u])

@dataclass(frozen=True, slots=True)
class Options:
    preprocess: bool = True
    symmetry: bool = True
    cuts_one_cluster: bool = True
    cuts_two_clusters: bool = True
    min_size: int | None = None  # tamaño minimo de cluster activo para el SCP capacitado
    max_size: int | None = None  # tamaño máximo de cluster activo para el SCP capacitado
    MAX_K: int | None = None  # número máximo de clústeres permitidos
    exact_K: bool = False  # si True y MAX_K != None, fuerza exactamente MAX_K clusters (MILP_3, ec. 57)

@dataclass(frozen=True, slots=True)
class MILPData:
    reduction: Reduction
    options: Options
    K: range
    Z: tuple[int, ...]
    X: tuple[tuple[int, int],...]
    T: tuple[tuple[int, int, int],...]
    symmetry_51: tuple[tuple[int, int],...]
    symmetry_52: tuple[tuple[int, int], ...]
    cuts_48: tuple[tuple[int, int],...]
    cuts_47: tuple[tuple[int, int, int],...]
    node_weights: tuple[int, ...]
    @property
    def instance(self) -> Instance:
        return self.reduction.reduced

    def ties(self) -> tuple[int, ...]:
        return self.reduction.ties

    @classmethod
    def from_instance(cls, original: Instance, options: Options = Options()) -> 'MILPData':
        if options.min_size is not None and options.min_size < 1:
            raise ValueError("min_size debe ser >= 1")

        if options.min_size is not None and options.max_size is not None and options.min_size > options.max_size:
            raise ValueError("min_size no puede ser mayor que max_size")


        network = reduce_sheets(original) if options.preprocess else without_reducing(original)

        leaf_count = {orig_u: 1 for orig_u in network.a_original}
        for _, neighbor in network.sheets:
            leaf_count[neighbor] += 1

        weights = tuple(leaf_count[orig_u] for orig_u in network.a_original)

        if options.max_size is not None and any(w > options.max_size for w in weights):
            network = without_reducing(original)
            weights = (1,) * original.n

        instance = network.reduced
        n_clusters = instance.n if options.MAX_K is None else min(instance.n, options.MAX_K)

        if n_clusters < 1:
            raise ValueError("MAX_K debe ser >= 1")

        K = range(n_clusters)
        symmetry = options.symmetry
        has_capacity = options.min_size is not None or options.max_size is not None
        Z = tuple(K) if has_capacity else ()

        X = tuple(
            (u, k)
            for u in range(instance.n)
            for k in (range(min(u + 1, n_clusters)) if symmetry else K)
        )

        T = tuple(
            (u, v, k)
            for u, v, _ in instance.edges
            for k in (range(min(u + 1, n_clusters)) if symmetry else K)
        )

        # Corte (51) solo es válido si |K| == n y no hay cota minima >= 2;
        # en caso contrario se usa el corte (52) (Köhler et al., 2013, Secc. 5).
        use_51 = symmetry and n_clusters == instance.n and (options.min_size or 0) <= 1
        s51 = tuple((u, k) for u in range(instance.n) for k in range(1, u)) if use_51 else ()
        s52 = (
            tuple((j, k) for k in range(2, n_clusters) for j in range(k, instance.n))
            if (symmetry and not use_51)
            else ()
        )
        neighbors = instance.neighbors()
        candidates = [u for u in range(instance.n) if neighbors[u] and network.ties[u] == 0]

        c48 = tuple(
            (u, w)
            for u in candidates
            for w in K
            if options.cuts_one_cluster and everyone_can(u, w, symmetry, neighbors)
        )

        c47 = tuple(
            (u, w, w2)
            for u in candidates
            for w2 in K
            for w in range(w2)
            if options.cuts_two_clusters and everyone_can(u, w, symmetry, neighbors)
        )

        return cls(network, options, K, Z, X, T, s51, s52, c48, c47, weights)
