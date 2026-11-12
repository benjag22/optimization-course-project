"""MILP_2 + mejoras de la secc. 4 en Gurobi.

Uso:
    datos = DatosMilp2.desde(Instance.from_file("data/raw/mtunis"))
    modelo = Milp2Gurobi(datos)
    modelo.model.Params.TimeLimit = 3600
    modelo.model.optimize()
    assignation = modelo.assignation()   # en índices de la instancia ORIGINAL
"""
import gurobipy as gp
from gurobipy import GRB
from src.models.MILP import MILPData

class MILPGurobi:
    def __init__(self, data: MILPData, env: gp.Env | None = None) -> None:
        self.data = data
        self.model = gp.Model(f"milp_{data.instance.name}", env=env)
        self._variables()
        self._objective()
        self._base_restrictions()
        self._improvements()
        self.model.update()

    # vars:(42)-(46):
    def _variables(self) -> None:
        d, m = self.data, self.model
        self.x = m.addVars(d.X, vtype=GRB.BINARY, name="x")
        self.t = m.addVars(d.T, lb=0, ub=1, name="t")
        self.s = m.addVars(d.X, lb=0, ub=1, name="s")
        self.r = m.addVars(d.K, lb=0, ub=1, name="r")

    def _x(self, u: int, k: int) -> gp.LinExpr:
        var = self.x.get((u, k))
        return gp.LinExpr(var) if var is not None else gp.LinExpr()

    # funcion objetivo:
    def _objective(self) -> None:
        self.model.setObjective(self.r.sum(), GRB.MAXIMIZE)

    #  restricciones de 3.3:
    def _base_restrictions(self) -> None:
        d, m, x, t, s, r = self.data, self.model, self.x, self.t, self.s, self.r
        V = range(d.instance.n)

        # (10) con la simetria en (50) porque x_uk solo existe para k <= u
        m.addConstrs((x.sum(u, "*") == 1 for u in V), name="c10_50")

        # (13)-(15): t_uvk = x_uk · x_vk
        m.addConstrs((t[u, v, k] <= x[u, k] for u, v, k in d.T), name="c13")
        m.addConstrs((t[u, v, k] <= x[v, k] for u, v, k in d.T), name="c14")
        m.addConstrs((t[u, v, k] >= x[u, k] + x[v, k] - 1 for u, v, k in d.T), name="c15")

        # (37): un cluster vacio no importa:
        m.addConstrs((r[k] <= x.sum("*", k) for k in d.K), name="c37")

        # (38): auto-lazo de peso L en v aporta 2L·s_vk a la izquierda y 2L·x_vk a la derecha:
        for k in d.K:
            izq, der = gp.LinExpr(), gp.LinExpr()
            for u, v, c in d.instance.edges:
                if (u, k) in s:
                    izq.add(s[u, k], c)
                if (v, k) in s:
                    izq.add(s[v, k], c)
                if (u, v, k) in t:
                    der.add(t[u, v, k], 2 * c)
            for v, lazo in enumerate(d.ties()):
                if lazo and (v, k) in s:
                    izq.add(s[v, k], 2 * lazo)
                    der.add(x[v, k], 2 * lazo)
            m.addConstr(izq == der, name=f"c38[{k}]")

        # (39)-(41):
        m.addConstrs((s[u, k] <= r[k] for u, k in d.X), name="c39")
        m.addConstrs((s[u, k] <= x[u, k] for u, k in d.X), name="c40")
        m.addConstrs((s[u, k] >= r[k] + x[u, k] - 1 for u, k in d.X), name="c41")

    # mejoras de las seccion 4):
    def _improvements(self) -> None:
        d, m, x = self.data, self.model, self.x
        nb = d.instance.neighbors()

        # (51):
        m.addConstrs((x[u, k] <= x[k, k] for u, k in d.symmetry_51), name="c51")

        # (48):
        m.addConstrs(
            (self._x(u, w) >= gp.quicksum(self._x(v, w) for v in nb[u]) - (len(nb[u]) - 1)
             for u, w in d.cuts_48),
            name="c48",
        )

        # (47):
        m.addConstrs(
            (1 - self._x(u, w) - self._x(u, w2)
             + gp.quicksum(self._x(v, w) + self._x(v, w2) for v in nb[u]) <= len(nb[u])
             for u, w, w2 in d.cuts_47),
            name="c47",
        )

    # solucion
    def assignation(self) -> list[int]:
        if self.model.SolCount == 0:
            raise RuntimeError("El modelo no tiene solucion entera")
        red = [-1] * self.data.instance.n
        for (u, k), var in self.x.items():
            if var.X > 0.5:
                red[u] = k
        return self.data.reduction.expand(red)