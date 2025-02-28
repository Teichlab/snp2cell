from contextlib import contextmanager
from io import BytesIO
import pytest
import numpy as np
import pandas as pd
import networkx as nx
import scanpy as sc
from matplotlib import pyplot as plt
import snp2cell


@contextmanager
def plot_nonempty(format="png"):
    buf = BytesIO()
    plt.figure()
    plt.savefig(buf, format=format)
    buf.seek(0)
    empty_plot_size = buf.getbuffer().nbytes
    buf.close()

    buf = BytesIO()
    yield buf
    buf.seek(0)
    assert buf.getbuffer().nbytes > empty_plot_size, "Plot should not be empty"
    buf.close()


@pytest.fixture
def snp2cell_instance():
    return snp2cell.SNP2CELL(seed=42)


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
