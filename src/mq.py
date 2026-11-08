import collections
from collections.abc import Sequence
from src.instance import Instance

def mq(inst: Instance, assignation: Sequence[int]) -> float:
    intra: collections.Counter[int] = collections.Counter()
    inter: collections.Counter[int] = collections.Counter()
    for u, v, c in inst.edges:
        a, b = assignation[u], assignation[v]
        if a == b:
            intra[a] += c
        else:
            inter[a] += c
            inter[b] += c
    return sum(2 * intra[k] / (2 * intra[k] + inter[k]) for k in intra)