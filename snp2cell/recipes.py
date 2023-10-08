import logging
import os
import typing
from pathlib import Path

import pandas as pd

import snp2cell
from snp2cell import SNP2CELL
from snp2cell.util import add_logger


@add_logger()
def filter_summ_stat_file(
    s2c_obj: typing.Union[str, os.PathLike],
    summ_stat_bed: typing.Union[str, os.PathLike],
    ld_score_bed: typing.Union[str, os.PathLike],
    out_name: typing.Union[str, os.PathLike],
    summ_stat_header: typing.Optional[str] = None,
    ld_score_header: typing.Optional[str] = None,
    pos2gene_csv: typing.Optional[typing.Union[str, os.PathLike]] = None,
    log: logging.Logger = logging.getLogger(),
) -> None:
    """
    Filter a file with summary statistics by the genomic locations of network nodes in s2c object and merge
    with linkage-disequilibrium scores (l2) from a second file. The resulting file can be used as an input
    for `score_snp`.

    `summ_stat_bed` needs to be a bed style tsv file with a header and at least these columns:
    ["Chromosome", "Start", "End", "Beta", "SE"]

    `ld_score_bed` needs to be a bed style tsv file with a header and at least these columns:
    ["Chromosome", "Position", "L2"]
    """
    # load object
    log.info("load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # get network locations
    gene2pos = None
    if pos2gene_csv:
        log.info(f"get pos2gene mapping from file: {Path(pos2gene_csv).resolve()}")
        gene2pos = pd.read_csv(pos2gene_csv, header=None, index_col=1)[0].to_dict()

    log.info("extract genomic locations for graph nodes")
    nx_loc_df = snp2cell.util.graph_nodes_to_bed(s2c.grn, gene2pos=gene2pos)
    assert nx_loc_df is not None

    # filter summ stats
    log.info("filter summary statistics by node locations")
    summ_stat_kwargs = {}
    if summ_stat_header:
        summ_stat_kwargs = dict(
            names=summ_stat_header.split(","),
            header=None,
        )

    snp_filt_df = snp2cell.util.filter_summ_stat(
        summ_stat_path=summ_stat_bed,
        network_loc=nx_loc_df,
        summ_stat_kwargs=summ_stat_kwargs,
    )
    assert snp_filt_df is not None

    # add LD scores
    log.info("add LD score locations and save")
    add_df_kwargs: dict[str, typing.Union[int, list[str], None]] = {}
    if ld_score_header:
        add_df_kwargs["names"] = ld_score_header.split(",")
        add_df_kwargs["header"] = None

    snp2cell.util.add_col_to_bed(
        snp_filt_df,
        ld_score_bed,
        add_df_merge_on=["Chromosome", "Position", "Position"],
        add_df_kwargs=add_df_kwargs,
        out_path=out_name,
        return_df=False,
    )
