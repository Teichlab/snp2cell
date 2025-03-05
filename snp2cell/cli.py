import logging
import os
import pickle
import typing
from enum import Enum
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import pandas as pd
import scanpy as sc
import typer  # type: ignore
from typing_extensions import Annotated
import networkx as nx

import snp2cell
from snp2cell.recipes import filter_summ_stat_file
from snp2cell.snp2cell_class import SNP2CELL
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
    log.info("create SNP2CELL object")
    s2c = SNP2CELL()
    s2c.add_grn_from_networkx(G)

    # save
    log.info(f"save SNP2CELL object to {Path(out_name).resolve()}")
    s2c.save_data(out_name)


@app.command()
@add_logger()
def create_gene2pos_mapping(
    pos2gene_csv: Annotated[
        Path,
        typer.Argument(help="output path for csv file with mapping"),
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
    # query biomart to obtain genomic locations for genes
    log.info(f"query biomart host: {host}")
    pos2gene = snp2cell.util.get_gene2pos_mapping(host=host, rev=True)

    # save to file
    log.info(f"save to file: {Path(pos2gene_csv).resolve()}")
    pd.DataFrame(pos2gene, index=[0]).T.sort_values(0).to_csv(  # type: ignore
        pos2gene_csv, header=False
    )


def filter_summ_stats(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    summ_stat_bed: Annotated[
        Path,
        typer.Argument(help="path to tsv file with summary statistics"),
    ],
    ld_score_bed: Annotated[
        Path, typer.Argument(help="path to tsv file with ld scores")
    ],
    out_name: Annotated[
        Path,
        typer.Argument(help="output path for filtered summary statistics"),
    ],
    summ_stat_header: Annotated[
        typing.Optional[str], typer.Option(help="comma separated string with header")
    ] = None,
    ld_score_header: Annotated[
        typing.Optional[str], typer.Option(help="comma separated string with header")
    ] = None,
    pos2gene_csv: Annotated[
        typing.Optional[Path],
        typer.Option(
            "--pos2gene",
            "-p",
            help="csv file with no header and location (chrX:XXX-XXX) to gene symbol mapping; default: retrieve from biomart",
        ),
    ] = None,
):
    """
    Filter a file with summary statistics by the genomic locations of network nodes in s2c object and merge
    with linkage-disequilibrium scores (l2) from a second file.

    `summ_stat_bed` needs to be a bed style tsv file with a header and at least these columns:
    ["Chromosome", "Start", "End", "Beta", "SE"]

    `ld_score_bed` needs to be a bed style tsv file with a header and at least these columns:
    ["Chromosome", "Position", "L2"]
    """
    filter_summ_stat_file(
        s2c_obj,
        summ_stat_bed,
        ld_score_bed,
        out_name,
        summ_stat_header,
        ld_score_header,
        pos2gene_csv,
    )


@app.command()
@add_logger()
def export_locations(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    region_loc_path: Annotated[
        Path, typer.Option("--output", "-o", help="output path for regions")
    ] = "peak_locations.txt",
    pos2gene: Annotated[
        typing.Optional[Path],
        typer.Option(
            "--pos2gene",
            "-p",
            help="csv file with no header and location (chrX:XXX-XXX) to gene symbol mapping. If not provided, no mapping will be done. If a path is provided the mapping will be read from the file. If a URL is provided, the mapping will be queried from biomart.",
        ),
    ] = None,
    log=logging.getLogger(),
):
    """
    Save the genomic locations of network nodes in the s2c object to a tsv file.
    """
    # load object
    log.info("load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # translate gene names to locations
    if pos2gene is not None:
        if os.path.exists(pos2gene):
            log.info(f"get pos2gene mapping from file: {Path(pos2gene).resolve()}")
            pos2gene_dict = pd.read_csv(pos2gene, header=None, index_col=0)[1].to_dict()
            gene2pos_dict = {v: k for k, v in pos2gene_dict.items()}
        elif bool(urlparse(pos2gene).scheme and urlparse(pos2gene).netloc):
            log.info(f"query biomart for pos2gene mapping: '{pos2gene}'")
            gene2pos_dict = snp2cell.util.get_gene2pos_mapping(rev=False)
        else:
            log.error(
                f"pos2gene file '{pos2gene}' does not exist and is not a valid URL"
            )
            raise FileNotFoundError(pos2gene)
        log.info(f"translate genes to locations")
        s2c.grn = nx.relabel_nodes(s2c.grn, gene2pos_dict)

    # save
    log.info(f"save regions to {Path(region_loc_path).resolve()}")
    s2c.util.export_for_fgwas(s2c, region_loc_path=region_loc_path)


@app.command()
@add_logger()
def score_snp(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    fgwas_output_path: Annotated[
        Path,
        typer.Argument(
            help="path to tsv.gz file with SNP Bayes factors and weights per region calculated by nf-fgwas"
        ),
    ],
    region_loc_path: Annotated[
        Path,
        typer.Argument(
            help="path to tsv file with genomic locations of regions (result from `export_locations()`)"
        ),
    ],
    rbf_table_path: Annotated[
        Path,
        typer.Option(
            "--output-table",
            "-o",
            help="path for saving the Regional Bayes Factors (RBF) as a table. If not provided, the table will not be saved.",
        ),
    ] = None,
    save_key: Annotated[
        str, typer.Option("--save-key", "-k", help="name for saving scores in object")
    ] = "snp_score",
    lexpand: Annotated[
        int,
        typer.Option(
            "--lexpand",
            "-l",
            help="number of base pairs to expand the region to the left",
            default=250,
        ),
    ] = 250,
    rexpand: Annotated[
        int,
        typer.Option(
            "--rexpand",
            "-r",
            help="number of base pairs to expand the region to the right",
            default=250,
        ),
    ] = 250,
    pos2gene: Annotated[
        typing.Optional[Path],
        typer.Option(
            "--pos2gene",
            "-p",
            help="csv file with no header and location (chrX:XXX-XXX) to gene symbol mapping. If not provided, no mapping will be done. If a path is provided the mapping will be read from the file. If a URL is provided, the mapping will be queried from biomart.",
        ),
    ] = None,
    n_cpu: Annotated[
        typing.Optional[int], typer.Option(help="number of cpus to use")
    ] = None,
    log=logging.getLogger(),
):
    """
    Add fGWAS scores for network nodes based on GWAS summary statistics. Then propagate the scores across the network
    and calculate statistics based on random permutations. All calculated information will be saved in the s2c object.

    This assumes the nf-fgwas pipeline (https://github.com/cellgeni/nf-fgwas) has been run.
    The nf-fgwas output file should then be in a folder like `results/LDSC_results/<studyid>/input.gz`.
    The region location file (nf-fgwas input) can be generated with `export_for_fgwas()`.

    Calculated Regional Bayes Factors (RBF) can be saved to a table by setting --output-table.
    The columns in the output file correspond to: region ID, SNP log BF, SNP log weight including distance and LD score.

    For the scores added to the snp2cell object, the region locations are expanded to the full length of the region (`--lexpand` and `--rexpand`).
    This is assuming that each region was represented by its center for the fgwas analysis and that all regions have the same length (default 500bp).
    If this is not the case, you can use the returned data frame and expand the regions manually.
    """
    # load object
    log.info("load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    if not isinstance(s2c.grn, nx.Graph):
        msg = "GRN should be a networkx graph"
        log.error(msg)
        raise ValueError(msg)

    # load SNP scores
    log.info(f"load fgwas SNP BFs from {Path(fgwas_output_path).resolve()}")
    log.info(f"compute SNP scores")
    snp_scr, _ = snp2cell.util.load_fgwas_scores(
        fgwas_output_path=fgwas_output_path,
        region_loc_path=region_loc_path,
        rbf_table_path=rbf_table_path,
        num_cores=n_cpu,
    )
    log.info(f"computed scores for {len(snp_scr)} regions")
    log.info(f"top scores: \n{pd.Series(snp_scr).sort_values(ascending=False)[:5]}")

    # translate locations to genes
    if pos2gene is not None:
        if os.path.exists(pos2gene):
            log.info(f"get pos2gene mapping from file: {Path(pos2gene).resolve()}")
            pos2gene_dict = pd.read_csv(pos2gene, header=None, index_col=0)[1].to_dict()
        elif bool(urlparse(pos2gene).scheme and urlparse(pos2gene).netloc):
            log.info(f"query biomart for pos2gene mapping: '{pos2gene}'")
            pos2gene_dict = snp2cell.util.get_gene2pos_mapping(rev=True)
        else:
            log.error(
                f"pos2gene file '{pos2gene}' does not exist and is not a valid URL"
            )
            raise FileNotFoundError(pos2gene)
        log.info(f"translate locations to genes")
        snp_scr = {
            (pos2gene_dict[k] if k in pos2gene_dict else k): v
            for k, v in snp_scr.items()
        }

    # propagate score
    log.info("add scores to SNP2CELL object")
    s2c.add_score(snp_scr, score_key=save_key, num_cores=n_cpu)

    # save
    log.info(f"save SNP2CELL object to {Path(s2c_obj).resolve()}")
    s2c.save_data(s2c_obj)


@app.command()
@add_logger()
def add_score(
    s2c_obj: Annotated[Path, typer.Argument(help="path to SNP2CELL object")],
    score_file: Annotated[
        Path,
        typer.Argument(
            help="Path to tsv file with scores for network nodes. Assuming there is no header and the first column contains the node names, the second column the scores."
        ),
    ],
    save_key: Annotated[
        str, typer.Option("--save-key", "-k", help="name for saving scores in object")
    ] = "snp_score",
    pos2gene: Annotated[
        typing.Optional[Path],
        typer.Option(
            "--pos2gene",
            "-p",
            help="csv file with no header and location (chrX:XXX-XXX) to gene symbol mapping. If not provided, no mapping will be done. If a path is provided the mapping will be read from the file. If a URL is provided, the mapping will be queried from biomart.",
        ),
    ] = None,
    n_cpu: Annotated[
        typing.Optional[int], typer.Option(help="number of cpus to use")
    ] = None,
    log=logging.getLogger(),
):
    """
    Add scores for network nodes to the s2c object and propagate the scores across the network.
    """
    # load object
    log.info("load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # load scores
    log.info(f"load scores from {Path(score_file).resolve()}")
    scores = pd.read_csv(score_file, sep="\t", index_col=0, header=None).squeeze()
    log.info(f"top scores: \n{scores.sort_values(ascending=False)[:5]}")

    # translate locations to genes
    if pos2gene is not None:
        if os.path.exists(pos2gene):
            log.info(f"get pos2gene mapping from file: {Path(pos2gene).resolve()}")
            pos2gene_dict = pd.read_csv(pos2gene, header=None, index_col=0)[1].to_dict()
        elif bool(urlparse(pos2gene).scheme and urlparse(pos2gene).netloc):
            log.info(f"query biomart for pos2gene mapping: '{pos2gene}'")
            pos2gene_dict = snp2cell.util.get_gene2pos_mapping(rev=True)
        else:
            log.error(
                f"pos2gene file '{pos2gene}' does not exist and is not a valid URL"
            )
            raise FileNotFoundError(pos2gene)
        log.info(f"translate locations to genes")
        scores = {
            (pos2gene_dict[k] if k in pos2gene_dict else k): v
            for k, v in scores.items()
        }

    # propagate score
    log.info("add scores to SNP2CELL object")
    s2c.add_score(scores.to_dict(), score_key=save_key, num_cores=n_cpu)

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
        typing.Optional[str],
        typer.Option(
            "--save-key",
            "-k",
            help="name for saving scores in object; default: `(score_key1 - score_key2)`",
        ),
    ] = None,
    n_cpu: Annotated[
        typing.Optional[int], typer.Option(help="number of cpus to use")
    ] = None,
    log=logging.getLogger(),
):
    """
    Add a new score that is a contrast of two scores, propagate it across the network and calculate statistics
    based on random permutations.
    """
    # load object
    log.info("load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)

    # contrast scores
    scr1 = s2c.get_scores(which="original")[score_key1]  # .replace({np.nan: 0})
    scr2 = s2c.get_scores(which="original")[score_key2]  # .replace({np.nan: 0})
    scr: pd.Series = scr1 - scr2
    scr_dct = scr[(scr > 0) & ~scr.isna()].to_dict()
    log.info(f"top scores: \n{pd.Series(scr_dct).sort_values(ascending=False)[:5]}")

    # propagate score
    log.info("add scores to SNP2CELL object")
    if save_key is None:
        save_key = f"({score_key1} - {score_key2})"
    s2c.add_score(scr_dct, score_key=save_key, num_cores=n_cpu)

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
        typing.Optional[int],
        typer.Option("--n-cpu", "-c", help="number of cpus to use"),
    ] = None,
    log=logging.getLogger(),
):
    """
    Add an anndata object to the s2c object, find differentially expressed genes and propagate the gene scores across the network.
    Then the DE scores and previously computed SNP scores are combined and statistics are computed based on random permutations.
    """
    # load anndata
    log.info(f"load anndata from {Path(anndata).resolve()}")
    ad = sc.read_h5ad(anndata)
    if use_raw:
        log.info("load from `ad.raw` attribute")
        ad = ad.raw.to_adata()
    if run_lognorm:
        log.info("log-normalise")
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
    if "base" not in ad.uns["log1p"]:
        ad.uns["log1p"]["base"] = None

    # load object
    log.info("load SNP2CELL object")
    s2c = SNP2CELL(s2c_obj)
    s2c.link_adata(ad, overwrite=True)

    # score de
    log.info("add DE scores to SNP2CELL object")
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
    log.info("load SNP2CELL object")
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
