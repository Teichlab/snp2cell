import pandas as pd
from unittest.mock import patch, MagicMock
from snp2cell.util import get_gene2pos_mapping
import os
import networkx as nx
import pytest
import snp2cell
import numpy as np


@patch("snp2cell.util.pybiomart.Server")
def test_get_gene2pos_mapping(mock_server_class):
    # Prepare the mock data
    df_mock = pd.DataFrame(
        {
            "external_gene_name": ["geneA", "geneB"],
            "chromosome_name": ["1", "10"],
            "start_position": [100, 200],
            "end_position": [150, 250],
            "strand": [1, 1],
            "ensembl_gene_id": ["ENSG000001", "ENSG000002"],
        }
    )

    # Setup mocking
    mock_server = MagicMock()
    mock_mart = MagicMock()
    mock_dataset = MagicMock()

    mock_server_class.return_value = mock_server
    mock_server.__getitem__.return_value = mock_mart
    mock_mart.__getitem__.return_value = mock_dataset
    mock_dataset.query.return_value = df_mock

    # Call the function
    gene2pos = get_gene2pos_mapping()

    # Validate the returned dictionary
    assert "geneA" in gene2pos, "Expected 'geneA' in mapping"
    assert "geneB" in gene2pos, "Expected 'geneB' in mapping"

    assert gene2pos["geneA"] == "chr1:100-150", "Incorrect mapping for 'geneA'"
    assert gene2pos["geneB"] == "chr10:200-250", "Incorrect mapping for 'geneB'"


def test_export_for_fgwas(snp2cell_instance, tmp_path):
    G = nx.DiGraph()
    G.add_edge("geneA", "chr1:100-200")
    G.add_edge("geneA", "chr6:200-200")
    G.add_edge("geneA", "chr10:310-427")
    snp2cell_instance.add_grn_from_networkx(G)

    region_path = tmp_path / "peak_locations.txt"
    compressed_path = tmp_path / "peak_locations.txt.gz"

    snp2cell.util.export_for_fgwas(snp2cell_instance, region_loc_path=str(region_path))

    # Check that the plain and compressed files exist
    assert region_path.exists(), "Region location file should be created"
    assert compressed_path.exists(), "Compressed region file should be created"

    # Read the exported file and verify expected columns
    df = pd.read_csv(compressed_path, sep="\t", header=None)
    expected_df = pd.DataFrame(
        [
            [1, 150],
            [6, 200],
            [10, 368],
        ]
    )
    pd.testing.assert_frame_equal(df, expected_df)


def test_load_fgwas_scores(snp2cell_instance, tmp_path):
    # Create a temporary fgwas output file with two rows.
    fgwas_output_path = tmp_path / "fgwas_output.txt"
    with open(fgwas_output_path, "w") as f:
        f.write(f"0\t{np.log(2)}\t0\n1\t{np.log(3)}\t0\n")

    # Create a temporary region location file with header (as in export_for_fgwas).
    region_loc_path = tmp_path / "region_loc.txt"
    # Two regions with hm_chr and hm_pos to match indices 0 and 1.
    region_df = pd.DataFrame({"hm_chr": [1, 2], "hm_pos": [150, 250]})
    region_df.to_csv(region_loc_path, sep="\t", index=False)

    # Specify path for the output RBF table.
    rbf_table_path = tmp_path / "rbf_table.txt"

    # Call load_fgwas_scores.
    score_dct, _ = snp2cell.util.load_fgwas_scores(
        fgwas_output_path=str(fgwas_output_path),
        region_loc_path=str(region_loc_path),
        rbf_table_path=str(rbf_table_path),
        lexpand=50,
        rexpand=50,
    )

    # Expected region names based on region_loc:
    name0 = "chr1:100-200"
    name1 = "chr2:200-300"

    # Expected log-RBF scores: log(2) and log(3) respectively.
    np.testing.assert_almost_equal(score_dct[name0], np.log(2), decimal=5)
    np.testing.assert_almost_equal(score_dct[name1], np.log(3), decimal=5)

    # Verify that the RBF table file was created.
    assert os.path.exists(str(rbf_table_path)), "rbf_table file should be created"


def test_get_reg_srt_keys():
    reg = "chr5:123-456"
    result = snp2cell.util.get_reg_srt_keys(reg)
    assert result == (5, 123, 456), "Expected (5, 123, 456) for 'chr5:123-456'"


@pytest.mark.parametrize(
    "rename, drop, filter, expected_columns",
    [
        ({}, [], None, ["CHR", "START", "END", "value", "extra"]),
        (
            {"extra": "new_extra"},
            [],
            None,
            ["CHR", "START", "END", "value", "new_extra"],
        ),
        ({}, ["extra"], None, ["CHR", "START", "END", "value"]),
        ({}, [], ["CHR", "START", "END"], ["CHR", "START", "END", "value"]),
    ],
)
def test_add_col_to_bed(rename, drop, filter, expected_columns):
    # Create sample DataFrames for bed and add_df
    bed = pd.DataFrame(
        {
            "CHR": [1, 2, 3],
            "START": [100, 200, 300],
            "END": [150, 250, 350],
            "value": [10, 20, 30],
        }
    )
    add_df = pd.DataFrame(
        {
            "CHR": [1, 2, 3],
            "START": [100, 200, 300],
            "END": [150, 250, 350],
            "extra": [7, 8, 9],
        }
    )

    # Merge DataFrames using add_col_to_bed
    merged = snp2cell.util.add_col_to_bed(
        bed, add_df, rename=rename, drop=drop, filter=filter
    )

    # Check that merged DataFrame contains expected columns
    assert all(
        col in merged.columns for col in expected_columns
    ), f"Expected columns {expected_columns}"

    # Compare with an expected merge using pandas merge
    if rename:
        add_df = add_df.rename(columns=rename)
    if drop:
        add_df = add_df.drop(columns=drop)
    if filter:
        add_df = add_df[filter]

    expected = pd.merge(bed, add_df, on=["CHR", "START", "END"])
    pd.testing.assert_frame_equal(
        merged.reset_index(drop=True),
        expected.sort_values(["CHR", "START", "END"]).reset_index(drop=True),
    )
