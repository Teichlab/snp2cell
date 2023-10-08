import pytest
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc


@pytest.fixture(scope="session")
def fake_grn():
    G = nx.barbell_graph(5, 1)
    G = nx.relabel_nodes(G, {i: str(i) for i in G})
    return G


@pytest.fixture(scope="session")
def fake_snp_score():
    snp_score = {"6": 1, "7": 1, "8": 1, "3": 1, "5": 0}
    return snp_score


@pytest.fixture(scope="session")
def fake_adata(fake_grn):
    df = pd.DataFrame(
        0,
        index=[str(i) for i in np.arange(90)],
        columns=[str(i) for i in np.arange(len(fake_grn))],
    )
    df.iloc[:30, [0, 2, 3]] = 100
    df.iloc[30:60, [2, 5, 9]] = 100
    df.iloc[60:90, [8, 9, 10]] = 100

    ad = sc.AnnData(df)
    ad.obs["cell type"] = ["A"] * 30 + ["B"] * 30 + ["C"] * 30

    sc.pp.normalize_total(ad)
    sc.pp.log1p(ad)
    ad.raw = ad
    sc.pp.scale(ad)

    return ad
