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

@dataclass(frozen=True, slots=True)
class MILPData:
    reduction: Reduction
    options: Options
    K: range
    X: tuple[tuple[int, int],...]
    T: tuple[tuple[int, int, int],...]
    symmetry_51: tuple[tuple[int, int],...]
    cuts_48: tuple[tuple[int, int],...]
    cuts_47: tuple[tuple[int, int, int],...]

    @property
    def instance(self) -> Instance:
        return self.reduction.reduced

    def ties(self) -> tuple[int, ...]:
        return self.reduction.ties

    @classmethod
    def from_instance(cls, original: Instance, options: Options = Options()) -> 'MILPData':
        network = reduce_sheets(original) if options.preprocess else without_reducing(original)
        instance = network.reduced
        K = range(instance.n)
        symmetry = options.symmetry

        X = tuple((u,k) for u in range(instance.n) for k in (range(u+1) if symmetry else K))
        T = tuple((u,v,k) for u,v,_ in instance.edges for k in (range(u+1) if symmetry else K))

        s51 = tuple((u, k) for u in range(instance.n) for k in range(1, u)) if symmetry else ()

        neighbors = instance.neighbors()
        candidates = [u for u in range(instance.n) if neighbors[u] and network.ties[u] == 0]

        c48 = tuple((u, w) for u in candidates for w in K
                    if options.cuts_one_cluster and everyone_can(u, w, symmetry, neighbors))

        c47 = tuple((u, w, w2) for u in candidates for w2 in K for w in range(w2)
                    if options.cuts_two_clusters and everyone_can(u, w, symmetry, neighbors))

        return cls(network, options, K, X, T, s51, c48, c47)

