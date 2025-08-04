# tools/grafo_const.py
"""
Crea un grafo ligero (NetworkX) con los artículos de la Constitución
conectados por capítulo / título. El grafo se construye en memoria
cada vez que arranca el agente y NO se persiste a disco.
"""

from pathlib import Path
import pandas as pd
import networkx as nx
from config import DATA_DIR

def build_const_graph(csv_path: Path | str = DATA_DIR / "constitucion.csv") -> nx.Graph:
    df = pd.read_csv(csv_path)
    G  = nx.Graph()

    for _, row in df.iterrows():
        try:
            art = int(row["ArticuloNo"])
        except (KeyError, ValueError):
            continue  # fila corrupta

        key = (row.get("Titulo"), row.get("Capitulo"))
        if pd.isna(key[0]) or pd.isna(key[1]):
            continue

        # todos los artículos que comparten Título y Capítulo son "vecinos"
        grupo = df[
            (df["Titulo"] == key[0]) &
            (df["Capitulo"] == key[1])
        ]["ArticuloNo"].astype(int)

        for vecino in grupo:
            if vecino != art:
                G.add_edge(art, int(vecino))

    return G

GRAPH_CONST = build_const_graph()
