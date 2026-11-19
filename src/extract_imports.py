"""Extrae el grafo de dependencias (imports) de un proyecto Python en el formato de data/.

Uso (desde la raiz de este proyecto):
    python -m src.extract_imports RUTA_PROYECTO [--salida data/mi_proyecto.txt]
                                  [--componente-mayor] [--excluir DIR ...]

Cada linea de salida es "origen destino" o "origen destino peso", donde peso es el
numero de sentencias import de origen que apuntan a destino (solo se escribe si es > 1).
Los nodos son modulos del proyecto con nombre punteado relativo a RUTA_PROYECTO
(un paquete se nombra por su __init__.py). Los imports a librerias externas y los
auto-imports se descartan.
"""
import argparse
import ast
import collections
import sys
from pathlib import Path

DEFAULT_EXCLUDED = {
    ".git", ".hg", ".venv", "venv", "env", ".env", "__pycache__", "node_modules",
    "build", "dist", "site-packages", ".tox", ".mypy_cache", ".pytest_cache", ".idea",
}


def module_name(path: Path, root: Path) -> tuple[str, bool]:
    parts = list(path.relative_to(root).with_suffix("").parts)
    is_package = parts[-1] == "__init__"
    if is_package:
        parts.pop()
    return ".".join(parts), is_package


def find_files(root: Path, excluded: set[str]) -> list[Path]:
    return sorted(
        p for p in root.rglob("*.py")
        if not any(part in excluded for part in p.relative_to(root).parts[:-1])
    )


def build_index(files: list[Path], root: Path) -> tuple[dict[str, str], dict[Path, tuple[str, bool]]]:
    """alias -> nombre canonico, y archivo -> (nombre, es_paquete)."""
    alias: dict[str, str] = {}
    info: dict[Path, tuple[str, bool]] = {}
    for p in files:
        name, is_pkg = module_name(p, root)
        if not name:  # __init__.py en la raiz
            continue
        info[p] = (name, is_pkg)
        alias[name] = name
    # Diseño src/: "import paquete" tambien resuelve a "src.paquete"
    for name in list(alias):
        head, _, rest = name.partition(".")
        if head == "src" and rest:
            alias.setdefault(rest, name)
    return alias, info


def resolve(candidate: str, alias: dict[str, str]) -> str | None:
    """Modulo interno mas largo que sea prefijo de candidate."""
    parts = candidate.split(".")
    for i in range(len(parts), 0, -1):
        hit = alias.get(".".join(parts[:i]))
        if hit is not None:
            return hit
    return None


def imports_of(tree: ast.AST, current: str, is_pkg: bool, alias: dict[str, str]) -> list[str]:
    package = current if is_pkg else current.rpartition(".")[0]
    targets: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                t = resolve(a.name, alias)
                if t:
                    targets.append(t)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                parts = package.split(".") if package else []
                up = node.level - 1
                if up > len(parts):
                    continue
                base_parts = parts[: len(parts) - up]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            for a in node.names:
                full = f"{base}.{a.name}" if base else a.name
                t = alias.get(full) or (resolve(base, alias) if base else None)
                if t:
                    targets.append(t)
    return targets


def largest_component(edges: dict[tuple[str, str], int]) -> dict[tuple[str, str], int]:
    adj: dict[str, set[str]] = collections.defaultdict(set)
    for a, b in edges:
        adj[a].add(b)
        adj[b].add(a)
    seen: set[str] = set()
    best: set[str] = set()
    for s in adj:
        if s in seen:
            continue
        seen.add(s)
        stack, comp = [s], {s}
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    stack.append(v)
        if len(comp) > len(best):
            best = comp
    return {(a, b): w for (a, b), w in edges.items() if a in best and b in best}


def extract(root: Path, excluded: set[str]) -> dict[tuple[str, str], int]:
    files = find_files(root, excluded)
    alias, info = build_index(files, root)
    edges: collections.Counter[tuple[str, str]] = collections.Counter()
    for p, (name, is_pkg) in info.items():
        try:
            tree = ast.parse(p.read_bytes(), filename=str(p))
        except (SyntaxError, ValueError) as e:
            print(f"OMITIDO {p}: {e}", file=sys.stderr)
            continue
        for target in imports_of(tree, name, is_pkg, alias):
            if target != name:
                edges[(name, target)] += 1
    return dict(edges)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("ruta", help="directorio raiz del proyecto Python")
    parser.add_argument("--salida", help="archivo de salida (por defecto data/<nombre>.txt)")
    parser.add_argument("--componente-mayor", action="store_true",
                        help="conserva solo la componente conexa mas grande")
    parser.add_argument("--excluir", nargs="*", default=[],
                        help="nombres de directorios extra a ignorar (ej. tests)")
    args = parser.parse_args()

    root = Path(args.ruta).resolve()
    if not root.is_dir():
        sys.exit(f"{root} no es un directorio")
    edges = extract(root, DEFAULT_EXCLUDED | set(args.excluir))
    if args.componente_mayor:
        edges = largest_component(edges)
    if not edges:
        sys.exit("no se encontraron dependencias internas")

    out = Path(args.salida) if args.salida else Path("data") / f"{root.name}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for (a, b), w in sorted(edges.items()):
            f.write(f"{a} {b}\n" if w == 1 else f"{a} {b} {w}\n")

    nodes = {v for e in edges for v in e}
    undirected = {tuple(sorted(e)) for e in edges}
    print(f"{out}: n={len(nodes)} m={len(undirected)}")


if __name__ == "__main__":
    main()
