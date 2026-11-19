"""MILP_2 + mejoras de la secc. 4 en CPLEX (docplex.mp).

Espeja MILPGurobi restriccion por restriccion, ambos leen las mismas listas de MILPData.

Uso:
    datos = MILPData.from_instance(Instance.from_file("data/A_s101.txt"))
    modelo = MILPCplex(datos)
    modelo.model.parameters.timelimit = 3600
    modelo.model.solve()
    assignation = modelo.assignation()   # en indices de la instancia ORIGINAL
"""
from docplex.mp.model import Model
from docplex.mp.linear import LinearExpr
from src.models.MILP import MILPData


class MILPCplex:
    def __init__(self, data: MILPData) -> None:
        self.data = data
        self.model = Model(name=f"milp_{data.instance.name}")
        self._index()
        self._variables()
        self._objective()
        self._base_restrictions()
        self._improvements()

    def _index(self) -> None:
        d = self.data
        self._ks_of_u: dict[int, list[int]] = {u: [] for u in range(d.instance.n)}
        self._us_of_k: dict[int, list[int]] = {k: [] for k in d.K}
        for u, k in d.X:
            self._ks_of_u[u].append(k)
            self._us_of_k[k].append(u)

    # vars:(42)-(46):
    def _variables(self) -> None:
        d, m = self.data, self.model
        self.x = m.binary_var_dict(list(d.X), name="x")
        self.t = m.continuous_var_dict(list(d.T), lb=0, ub=1, name="t")
        self.s = m.continuous_var_dict(list(d.X), lb=0, ub=1, name="s")
        self.r = m.continuous_var_dict(list(d.K), lb=0, ub=1, name="r")
        self.z = m.binary_var_dict(list(d.Z), name="z") if d.Z else None

    def _x(self, u: int, k: int) -> LinearExpr:
        var = self.x.get((u, k))
        return var if var is not None else self.model.linear_expr()

    # funcion objetivo:
    def _objective(self) -> None:
        self.model.maximize(self.model.sum(self.r[k] for k in self.data.K))

    #  restricciones de 3.3:
    def _base_restrictions(self) -> None:
        d, m, x, t, s, r = self.data, self.model, self.x, self.t, self.s, self.r
        V = range(d.instance.n)

        # (10) con la simetria en (50) porque x_uk solo existe para k <= u
        m.add_constraints(m.sum(x[u, k] for k in self._ks_of_u[u]) == 1 for u in V)

        # (13)-(15): t_uvk = x_uk · x_vk
        m.add_constraints(t[u, v, k] <= x[u, k] for u, v, k in d.T)
        m.add_constraints(t[u, v, k] <= x[v, k] for u, v, k in d.T)
        m.add_constraints(t[u, v, k] >= x[u, k] + x[v, k] - 1 for u, v, k in d.T)

        # (37): un cluster vacio no importa:
        m.add_constraints(r[k] <= m.sum(x[u, k] for u in self._us_of_k[k]) for k in d.K)

        # (38): auto-lazo de peso L en v aporta 2L·s_vk a la izquierda y 2L·x_vk a la derecha:
        for k in d.K:
            izq: list[tuple] = []
            der: list[tuple] = []
            for u, v, c in d.instance.edges:
                if (u, k) in s:
                    izq.append((s[u, k], c))
                if (v, k) in s:
                    izq.append((s[v, k], c))
                if (u, v, k) in t:
                    der.append((t[u, v, k], 2 * c))
            for v, lazo in enumerate(d.ties()):
                if lazo and (v, k) in s:
                    izq.append((s[v, k], 2 * lazo))
                    der.append((x[v, k], 2 * lazo))
            m.add_constraint(
                m.sum(c * var for var, c in izq) == m.sum(c * var for var, c in der)
            )

        # (39)-(41):
        m.add_constraints(s[u, k] <= r[k] for u, k in d.X)
        m.add_constraints(s[u, k] <= x[u, k] for u, k in d.X)
        m.add_constraints(s[u, k] >= r[k] + x[u, k] - 1 for u, k in d.X)

    # mejoras de las seccion 4) y extensiones (capacidad / K-SCP):
    def _improvements(self) -> None:
        d, m, x = self.data, self.model, self.x
        nb = d.instance.neighbors()

        # (51) o (52) según corresponda:
        m.add_constraints(x[u, k] <= x[k, k] for u, k in d.symmetry_51)
        m.add_constraints(
            x[j, k] <= m.sum(x[i, k - 1] for i in range(k - 1, j))
            for j, k in d.symmetry_52
        )

        # (48):
        m.add_constraints(
            self._x(u, w) >= m.sum(self._x(v, w) for v in nb[u]) - (len(nb[u]) - 1)
            for u, w in d.cuts_48
        )

        # (47):
        m.add_constraints(
            1 - self._x(u, w) - self._x(u, w2)
            + m.sum(self._x(v, w) + self._x(v, w2) for v in nb[u])
            <= len(nb[u])
            for u, w, w2 in d.cuts_47
        )

        # (57) forzar exactamente MAX_K clusters activos
        if d.options.exact_K and d.options.MAX_K is not None:
            m.add_constraints(m.sum(x[u, k] for u in self._us_of_k[k]) >= 1 for k in d.K)

        # SCP capacitado L * z_k <= tamaño(k) <= U * z_k
        if self.z is not None:
            self._add_min_and_max_clusters(d.options.min_size, d.options.max_size)

    def _add_min_and_max_clusters(self, min_size: int | None, max_size: int | None) -> None:
        d, m, x, z = self.data, self.model, self.x, self.z
        assert z is not None

        for k in d.K:
            members = self._us_of_k[k]
            cluster_size = m.sum(d.node_weights[u] * x[u, k] for u in members)
            if min_size is not None:
                m.add_constraint(cluster_size >= min_size * z[k])
            else:
                # si no hay cota minima, impedir que z_k = 1
                m.add_constraint(z[k] <= m.sum(x[u, k] for u in members))

            if max_size is not None:
                m.add_constraint(cluster_size <= max_size * z[k])
            else:
                # si no hay cota max, forzar z_k = 1
                m.add_constraints(z[k] >= x[u, k] for u in members)

    # solucion
    def assignation(self) -> list[int]:
        if not self.model.has_solution:
            raise RuntimeError("El modelo no tiene solucion entera")
        red = [-1] * self.data.instance.n
        for (u, k), var in self.x.items():
            if var.solution_value > 0.5:
                red[u] = k
        return self.data.reduction.expand(red)
