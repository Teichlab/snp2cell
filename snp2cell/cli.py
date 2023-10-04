import logging
import pickle
from enum import Enum
from pathlib import Path
from typing import List, Optional

import pandas as pd
import scanpy as sc
import typer
from typing_extensions import Annotated

import snp2cell
import snp2cell.util
from snp2cell import SNP2CELL
from snp2cell.util import add_logger

app = typer.Typer(pretty_exceptions_enable=True)


@app.command()
@add_logger()
def create_object(
    nx_graph: Annotated[Path, typer.Argument(help="path to pickled networkx graph")],
    out_name: Annotated[Path, typer.Argument(help="output path for SNP2CELL object")],
    log=logging.getLogger(),
):
    """
    Create an initial SNP2CELL object holding a network.
    This can be used as an input for other sub-commands.
    """
    # load networkx graph
    log.info(f"load networkx graph from {Path(nx_graph).resolve()}")
    with open(nx_graph, "rb") as f:
        G = pickle.load(f)

    # create object
    log.info(f"create SNP2CELL object")
    s2c = SNP2CELL()
    s2c.add_grn_from_networkx(G)

    # save
    log.info(f"save SNP2CELL object to {Path(out_name).resolve()}")
    s2c.save_data(out_name)


@app.command()
@add_logger()
def create_gene2pos_mapping(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    pos2gene_csv: Annotated[
        Path, typer.Argument(help="output path for csv file with mapping")
    ] = "pos2gene.csv",
    host: Annotated[
        str,
        typer.Option(
            "--biomart-host", "-h", help="biomart host to use; default: newest"
        ),
    ] = "http://www.ensembl.org",
    log=logging.getLogger(),
):
    """
    Query pybiomart to obtain genomic locations for genes in the network in the s2c object and save them to a csv file.
    """
    # load object
    log.info(f"load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # query biomart to obtain genomic locations for genes
    log.info(f"query biomart host: {host}")
    pos2gene = snp2cell.util.get_gene2pos_mapping(host=host, rev=True)

    # save to file
    log.info(f"save to file: {Path(pos2gene_csv).resolve()}")
    pd.DataFrame(pos2gene, index=[0]).T.sort_values(0).to_csv(
        pos2gene_csv, header=False
    )


@app.command()
@add_logger()
def filter_summ_stats(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    summ_stat_bed: Annotated[
        Path, typer.Argument(help="path to tsv file with summary statistics")
    ],
    ld_score_bed: Annotated[
        Path, typer.Argument(help="path to tsv file with ld scores")
    ],
    out_name: Annotated[
        Path, typer.Argument(help="output path for filtered summary statistics")
    ],
    summ_stat_header: Annotated[
        str, typer.Option(help="comma separated string with header")
    ] = None,
    ld_score_header: Annotated[
        str, typer.Option(help="comma separated string with header")
    ] = None,
    pos2gene_csv: Annotated[
        Path,
        typer.Option(
            "--pos2gene",
            "-p",
            help="csv file with no header and location (chrX:XXX-XXX) to gene symbol mapping; default: retrieve from biomart",
        ),
    ] = None,
    log=logging.getLogger(),
):
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
    log.info(f"load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # get network locations
    gene2pos = None
    if pos2gene_csv:
        log.info(f"get pos2gene mapping from file: {Path(pos2gene_csv).resolve()}")
        gene2pos = pd.read_csv(pos2gene_csv, header=None, index_col=1)[0].to_dict()

    log.info(f"extract genomic locations for graph nodes")
    nx_loc_df = snp2cell.util.graph_nodes_to_bed(s2c.grn, gene2pos=gene2pos)

    # filter summ stats
    log.info(f"filter summary statistics by node locations")
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

    # add LD scores
    log.info(f"add LD score locations and save")
    add_df_kwargs = {"index_col": 0}
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


@app.command()
@add_logger()
def score_snp(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    summ_stat_bed: Annotated[
        Path, typer.Argument(help="path to tsv file with summary statistics")
    ],
    save_key: Annotated[
        str, typer.Option("--save-key", "-k", help="name for saving scores in object")
    ] = "snp_score",
    pos2gene_csv: Annotated[
        Path,
        typer.Option(
            "--pos2gene",
            "-p",
            help="csv file with no header and location (chrX:XXX-XXX) to gene symbol mapping; default: retrieve from biomart",
        ),
    ] = None,
    n_cpu: Annotated[int, typer.Option(help="number of cpus to use")] = None,
    log=logging.getLogger(),
):
    """
    Calculate scores for network nodes based on GWAS summary statistics, propagate the scores across the network
    and calculate statistics based on random permutations. All calculated information will be saved in the s2c object.

    `summ_stat_bed` needs to be a bed style tsv file with a header and at least these columns:
    ["Chromosome", "Start", "End", "Beta", "SE", "L2"]
    This file can be created with `filter_summ_stats`.
    """
    # load object
    log.info(f"load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # load SNP scores
    log.info(f"load summary statistics from {Path(summ_stat_bed).resolve()}")
    regions = [n for n in s2c.grn.nodes if n[:3] == "chr"]
    log.info(f"compute SNP scores")
    snp_scr = snp2cell.util.get_snp_scores_parallel(
        regions, summ_stat_bed, num_cores=n_cpu
    )

    # translate locations to genes
    if pos2gene_csv:
        log.info(f"get pos2gene mapping from file: {Path(pos2gene_csv).resolve()}")
        pos2gene = pd.read_csv(pos2gene_csv, header=None, index_col=0)[1].to_dict()
    else:
        log.info(f"query biomart for pos2gene mapping")
        pos2gene = snp2cell.util.get_gene2pos_mapping(rev=True)
    snp_scr = {(pos2gene[k] if k in pos2gene else k): v for k, v in snp_scr.items()}
    log.info(f"top scores: \n{pd.Series(snp_scr).sort_values(ascending=False)[:5]}")

    # propagate score
    log.info(f"add scores to SNP2CELL object")
    s2c.add_score(snp_scr, score_key=save_key, num_cores=n_cpu)

    # save
    log.info(f"save SNP2CELL object to {Path(s2c_obj).resolve()}")
    s2c.save_data(s2c_obj)


@app.command()
@add_logger()
def contrast_scores(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    score_key1: Annotated[
        str, typer.Argument(help="key for scores stored in object (main)")
    ],
    score_key2: Annotated[
        str, typer.Argument(help="key for scores stored in object (reference)")
    ],
    save_key: Annotated[
        str,
        typer.Option(
            "--save-key",
            "-k",
            help="name for saving scores in object; default: `(score_key1 - score_key2)`",
        ),
    ] = None,
    n_cpu: Annotated[int, typer.Option(help="number of cpus to use")] = None,
    log=logging.getLogger(),
):
    """
    Add a new score that is a contrast of two scores, propagate it across the network and calculate statistics
    based on random permutations.
    """
    # load object
    log.info(f"load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # contrast scores
    scr1 = s2c.get_scores(which="original")[score_key1]  # .replace({np.nan: 0})
    scr2 = s2c.get_scores(which="original")[score_key2]  # .replace({np.nan: 0})
    scr = scr1 - scr2
    scr = scr[(scr > 0) & ~scr.isna()].to_dict()
    log.info(f"top scores: \n{pd.Series(scr).sort_values(ascending=False)[:5]}")

    # propagate score
    log.info(f"add scores to SNP2CELL object")
    if save_key is None:
        save_key = f"({score_key1} - {score_key2})"
    s2c.add_score(scr, score_key=save_key, num_cores=n_cpu)

    # save
    log.info(f"save SNP2CELL object to {Path(s2c_obj).resolve()}")
    s2c.save_data(s2c_obj)


class RankByChoice(str, Enum):
    abs = "abs"
    up = "up"
    down = "down"


@app.command()
@add_logger()
def score_de(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    anndata: Annotated[Path, typer.Argument(help="path to anndata object")],
    run_lognorm: Annotated[bool, typer.Option(help="log-normalise counts")] = False,
    use_raw: Annotated[bool, typer.Option(help="use `ad.raw` attribute")] = False,
    groupby: Annotated[
        str,
        typer.Argument(
            help="`ad.obs` column with annotation for computing differential expression"
        ),
    ] = "annot",
    groups: Annotated[
        Optional[List[str]],
        typer.Option(
            "--group",
            "-g",
            help="restrict to groups of `groupby`; may be set multiple times; default is to use all groups",
        ),
    ] = None,
    reference: Annotated[
        str,
        typer.Option(
            help="reference group to compare against; default is to compare against the rest"
        ),
    ] = "rest",
    method: Annotated[str, typer.Option(help="method for DE calculation")] = "wilcoxon",
    rank_by: Annotated[
        RankByChoice,
        typer.Option(
            "--rank-by",
            "-r",
            help="rank DE scores by absolute value, up- or downregulation; default: upregulation",
        ),
    ] = RankByChoice.up,
    snp_score_key: Annotated[
        str,
        typer.Option(
            "--snp-score-key", "-k", help="key for accessing saved snp scores in object"
        ),
    ] = "snp_score",
    n_cpu: Annotated[
        int, typer.Option("--n-cpu", "-c", help="number of cpus to use")
    ] = None,
    log=logging.getLogger(),
):
    """
    Add an anndata object to the s2c object, find differentially expressed genes and propagate the gene scores across the network.
    Then the DE scores and previously computed SNP scores are combined and statistics are computed based on random permutations.

    """
    # load anndata
    log.info(f"load anndata from {Path(anndata).resolve()}")
    ad = sc.read(anndata)
    if use_raw:
        log.info(f"load from `ad.raw` attribute")
        ad = ad.raw.to_adata()
    if run_lognorm:
        log.info(f"log-normalise")
        sc.pp.normalize_total(ad)
        sc.pp.log1p(ad)
    if ad.X.max() > 50:
        msg = "anndata does not seem to be log-normalised"
        log.error(msg)
        raise ValueError(msg)
    if ad.X.shape[1] < 10000:
        msg = "anndata does not seem to contain all genes"
        log.error(msg)
        raise ValueError(msg)

    # a bug in some anndata versions removes `base` entry
    if not "base" in ad.uns["log1p"]:
        ad.uns["log1p"]["base"] = None

    # load object
    log.info(f"load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)
    s2c.link_adata(ad, overwrite=True)

    # score de
    log.info(f"add DE scores to SNP2CELL object")
    s2c.adata_add_de_scores(
        groupby=groupby,
        num_cores=n_cpu,
        use_raw=False,
        method=method,
        groups=groups if groups else "all",
        reference=reference,
        rank_by=rank_by,
    )
    s2c.adata_combine_de_scores(
        group_key=groupby,
        score_key=snp_score_key,
        suffix="__zscore",
    )

    # save
    log.info(f"save SNP2CELL object to {Path(s2c_obj).resolve()}")
    s2c.save_data(s2c_obj)


@app.command()
@add_logger()
def combine_scores(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    groupby: Annotated[
        str, typer.Argument(help="`ad.obs` column with annotation used for DE scores")
    ] = "annot",
    snp_score_key: Annotated[
        str, typer.Argument(help="key for accessing saved snp scores in object")
    ] = "snp_score",
    log=logging.getLogger(),
):
    """
    Assuming that both a SNP score and DE scores have been added to the s2c object,
    combine SNP score with DE scores and compute statistics.
    """
    # load object
    log.info(f"load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # combine scores
    s2c.adata_combine_de_scores(
        group_key=groupby,
        score_key=snp_score_key,
        suffix="__zscore",
    )

    # save
    log.info(f"save SNP2CELL object to {Path(s2c_obj).resolve()}")
    s2c.save_data(s2c_obj)


if __name__ == "__main__":
    app()
