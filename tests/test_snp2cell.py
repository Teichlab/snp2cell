import pytest
import networkx as nx
import pandas as pd
import numpy as np
import snp2cell

snp2cell.util.set_num_cpu(1)


def test_initialization_with_path(snp2cell_instance, tmp_path):
    # Create a temporary file to simulate the path
    path = tmp_path / "test_data.pkl"
    snp2cell_instance.save_data(path=str(path))

    s2c = snp2cell.SNP2CELL(path=str(path), seed=42)
    assert s2c is not None, "snp2cell object was not created"
    assert s2c.grn is None, "GRN should be None"
    assert s2c.adata is None, "AnnData should be None"
    assert s2c.scores is None, "Scores should be None"


def test_init_scores(snp2cell_instance):
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    snp2cell_instance._set_grn(G)
    snp2cell_instance._init_scores()
    assert snp2cell_instance.scores is not None, "Scores should be initialized"
    assert (
        snp2cell_instance.scores_prop is not None
    ), "Propagated scores should be initialized"
    assert snp2cell_instance.scores_rand == {}, "Random scores should be initialized"
    assert snp2cell_instance.de_groups == {}, "DE groups should be initialized"


def test_set_grn(snp2cell_instance):
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    snp2cell_instance._set_grn(G)
    assert snp2cell_instance.grn is not None, "GRN should be set"
    assert list(snp2cell_instance.grn.edges) == [
        (1, 2),
        (2, 3),
    ], "GRN edges should match"


def test_add_de_groups(snp2cell_instance):
    snp2cell_instance._add_de_groups("group1", ["A", "B"])
    assert "group1" in snp2cell_instance.de_groups, "Group1 should be added"
    assert snp2cell_instance.de_groups["group1"] == [
        "A",
        "B",
    ], "Group1 values should match"

    with pytest.raises(ValueError):
        snp2cell_instance._add_de_groups("group1", ["C"])

    snp2cell_instance._add_de_groups("group2", ["C"])
    with pytest.raises(ValueError):
        snp2cell_instance._add_de_groups("group2", ["A"])


def test_get_perturbed_stats(snp2cell_instance):
    snp2cell_instance.scores_rand["test_key"] = pd.DataFrame(np.random.randn(10, 3))

    for suffix in snp2cell.SUFFIX:
        result = snp2cell_instance._get_perturbed_stats("test_key", suffix.value)
        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"


def test_robust_z_score():
    series = pd.Series([1, 2, 3, 4, 5])
    result = snp2cell.SNP2CELL._robust_z_score(series)
    assert isinstance(result, pd.Series), "Result should be a Series"
    assert len(result) == 5, "Result length should match input length"


def test_get_scores(snp2cell_instance):
    # Add some scores to the instance
    snp2cell_instance.add_grn_from_networkx(nx.from_edgelist([(1, 2), (2, 3)]))
    snp2cell_instance.scores = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    snp2cell_instance.scores_prop = pd.DataFrame({"A": [7, 8, 9], "B": [10, 11, 12]})
    snp2cell_instance.scores_rand = {"test_key": pd.DataFrame(np.random.randn(10, 3))}

    # Test retrieving original and propagated scores
    for which in ["original", "propagated"]:
        scores = snp2cell_instance.get_scores(which=which)
        assert scores is not None, "Scores should be retrieved"
        assert isinstance(scores, pd.DataFrame), "Scores should be a DataFrame"
        assert "A" in scores.columns, "Scores should have column 'A'"
        assert "B" in scores.columns, "Scores should have column 'B'"

    # Test retrieving perturbed scores
    scores = snp2cell_instance.get_scores(which="perturbed")
    assert scores is not None, "Scores should be retrieved"
    assert isinstance(scores, dict), "Scores should be a dictionary"
    assert "test_key" in scores, "Scores should have key 'test_key'"
    assert isinstance(scores["test_key"], pd.DataFrame), "Scores should be a DataFrame"

    # Test retrieving with query
    scores = snp2cell_instance.get_scores(which="propagated", query="A > 7")
    assert len(scores) == 2, "Query should filter the DataFrame"

    # Test retrieving with sort_key
    scores = snp2cell_instance.get_scores(which="propagated", sort_key="A")
    assert scores.iloc[0]["A"] == 9, "Scores should be sorted in descending order"


def test_remove_scores(snp2cell_instance):
    snp2cell_instance.add_grn_from_networkx(nx.from_edgelist([(1, 2), (2, 3)]))

    # Add some scores to the instance
    snp2cell_instance.scores = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    snp2cell_instance.scores_prop = pd.DataFrame(
        {"A": [7, 8, 9], "A__pval": [7, 8, 9], "B": [10, 11, 12]}
    )
    snp2cell_instance.scores_rand = {"A": pd.DataFrame(np.random.randn(10, 3))}

    # Test removing non-existing scores (should not raise an error)
    snp2cell_instance.remove_scores(which="original", items=["C"])
    assert snp2cell_instance.scores is not None, "Original scores should not be removed"
    assert (
        snp2cell_instance.scores.shape[1] == 2
    ), "Original scores should not be removed"
    assert (
        snp2cell_instance.scores_prop is not None
    ), "Propagated scores should not be removed"
    assert (
        snp2cell_instance.scores_prop.shape[1] == 3
    ), "Propagated scores should not be removed"
    assert "A" in snp2cell_instance.scores_rand, "Random scores should not be removed"

    # Test removing original scores
    snp2cell_instance.remove_scores(which="original", items=["A"])
    assert (
        "A" not in snp2cell_instance.scores.columns
    ), "Original scores should be removed"
    assert (
        "A" not in snp2cell_instance.scores_prop.columns
    ), "Propagated scores should also be removed"
    assert (
        "A__pval" not in snp2cell_instance.scores_prop.columns
    ), "Corresponding statistics should also be removed"
    assert (
        "A" not in snp2cell_instance.scores_rand
    ), "Random scores should also be removed"

    # Add scores to the instance
    snp2cell_instance.scores = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    snp2cell_instance.scores_prop = pd.DataFrame(
        {"A": [7, 8, 9], "A__pval": [7, 8, 9], "B": [10, 11, 12]}
    )
    snp2cell_instance.scores_rand = {"A": pd.DataFrame(np.random.randn(10, 3))}

    # Test removing propagated scores
    snp2cell_instance.remove_scores(which="propagated", items=["A"])
    assert (
        "A" in snp2cell_instance.scores.columns
    ), "Original scores should not be removed"
    assert (
        "A" not in snp2cell_instance.scores_prop.columns
    ), "Propagated scores should be removed"
    assert (
        "A" not in snp2cell_instance.scores_rand
    ), "Random scores should also be removed"

    # Add scores to the instance
    snp2cell_instance.scores = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    snp2cell_instance.scores_prop = pd.DataFrame(
        {"A": [7, 8, 9], "A__pval": [7, 8, 9], "B": [10, 11, 12]}
    )
    snp2cell_instance.scores_rand = {"A": pd.DataFrame(np.random.randn(10, 3))}

    # Test removing random scores
    snp2cell_instance.remove_scores(which="perturbed", items=["A"])
    assert (
        "A" in snp2cell_instance.scores.columns
    ), "Original scores should not be removed"
    assert (
        "A" in snp2cell_instance.scores_prop.columns
    ), "Propagated scores should not be removed"
    assert "A" not in snp2cell_instance.scores_rand, "Random scores should be removed"

    # Test removing all propagated scores
    snp2cell_instance.remove_scores(which="propagated")
    assert (
        snp2cell_instance.scores_prop.shape[1] == 0
    ), "All propagated scores should be removed"
    assert (
        len(snp2cell_instance.scores_rand) == 0
    ), "All random scores should also be removed"

    # Test removing all original scores
    snp2cell_instance.remove_scores(which="original")
    assert (
        snp2cell_instance.scores.shape[1] == 0
    ), "All original scores should be removed"
