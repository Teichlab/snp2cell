import snp2cell
import matplotlib.pyplot as plt

import snp2cell
from conftest import check_plot_nonempty

snp2cell.util.set_num_cpu(1)


def test_toy_example(fake_grn, fake_adata, fake_snp_score, tmp_path):
    s2c = snp2cell.SNP2CELL()
    s2c.add_grn_from_networkx(fake_grn)

    assert s2c is not None, "snp2cell object was not created"
    assert s2c.grn is not None, "networkx object was not added to snp2cell object"

    # add SNP score, propagate, calculate statistics
    s2c.add_score(score_dct=fake_snp_score, score_key="snp_score")

    assert (
        "snp_score" in s2c.scores
    ), "original snp_score was not added to snp2cell object"
    assert (
        "snp_score" in s2c.scores_prop
    ), "propagated snp_score was not added to snp2cell object"
    assert (
        "snp_score" in s2c.scores_rand
    ), "permuted snp_scores were not added to snp2cell object"
    assert (
        "snp_score__pval" in s2c.scores_prop
    ), "statistics were not calculated for snp_score"

    # add DE scores, propagate, calculate statistics
    s2c.link_adata(fake_adata)
    s2c.adata_add_de_scores(groupby="cell type", check=False)

    assert s2c.adata is not None, "anndata object was not added to snp2cell object"
    for grp in fake_adata.obs["cell type"].unique().tolist():
        assert (
            grp in s2c.de_groups["cell type"]
        ), f"cell type {grp} was not added to snp2cell object"
        assert (
            f"DE_{grp}__score" in s2c.scores
        ), f"original DE score for {grp} was not added to snp2cell object"
        assert (
            f"DE_{grp}__score" in s2c.scores_prop
        ), f"propagated DE score for {grp} was not added to snp2cell object"

    assert (
        f"DE_cell type__score" in s2c.scores_rand
    ), f"permuted DE scores for 'cell type' were not added to snp2cell object"

    for grp in fake_adata.obs["cell type"].unique().tolist():
        assert (
            f"DE_{grp}__score__pval" in s2c.scores_prop
        ), f"statistics were not calculated for {grp}"

    # combine each DE score with the SNP score
    s2c.adata_combine_de_scores(
        group_key="cell type", score_key="snp_score", suffix="__zscore"
    )

    assert (
        "min(DE_C__score__zscore,snp_score__zscore)" in s2c.scores_prop
    ), "DE score and SNP score were not combined"

    sum_a = s2c.scores_prop["min(DE_A__score__zscore,snp_score__zscore)"].sum()
    sum_b = s2c.scores_prop["min(DE_B__score__zscore,snp_score__zscore)"].sum()
    sum_c = s2c.scores_prop["min(DE_C__score__zscore,snp_score__zscore)"].sum()

    assert sum_c > sum_b > sum_a, "computed values are not correct"

    # test plotting
    plt.switch_backend("Agg")  # non-interactive backend (don't display plots)

    with check_plot_nonempty() as buf:
        s2c.plot_group_summary(score_key="snp_score")
        plt.savefig(buf, format="png")

    with check_plot_nonempty() as buf:
        s2c.plot_group_heatmap(score_key="snp_score", query="")
        plt.savefig(buf, format="png")

    # network plot only works with directed graphs at the moment
    # s2c.plot_network(score="snp_score", gene="1")

    # Close all plots to avoid memory issues
    plt.close("all")
