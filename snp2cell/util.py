import logging
import os
import pickle
import re
import typing
from functools import wraps
from inspect import signature
from pathlib import Path

import dill
import networkx as nx
import numpy as np
import pandas as pd
import pyranges
import scanpy
from joblib import Parallel, delayed
from tqdm import tqdm

import snp2cell

F = typing.TypeVar("F", bound=typing.Callable[..., typing.Any])


def set_num_cpu(n: int) -> None:
    """
    Globally set the number of CPUs.

    Parameters
    ----------
    n : int
        number of CPUs

    Returns
    -------
    """
    snp2cell.NCPU = n


def add_logger(show_start_end: bool = True) -> typing.Callable[[F], F]:
    """
    Decorator to add logging to a function or method.

    Parameters
    ----------
    show_start_end : bool
        display a message when entering and leaving the function or method. (Default value = True)

    Returns
    -------
    """

    def _add_logger(f):
        @wraps(f)
        def wrapped(*args, **kargs):
            func_name = f.__name__

            log = logging.getLogger(func_name)
            log.setLevel(logging.DEBUG)
            if not log.hasHandlers():
                c_handler = logging.StreamHandler()
                c_handler.setLevel(logging.INFO)
                c_format = logging.Formatter(
                    "[%(levelname)s - %(name)s - %(asctime)s]: %(message)s"
                )
                c_handler.setFormatter(c_format)
                log.addHandler(c_handler)

            if show_start_end:
                log.info(f"----- starting {func_name} -----")
                r = f(*args, **kargs, log=log)
                log.info(f"----- finished {func_name} -----")
            else:
                r = f(*args, **kargs, log=log)
            return r

        # edit signature to remove params (PEP-0362)
        remove_params = ["log"]
        sig = signature(f)
        sig = sig.replace(
            parameters=tuple(
                v for k, v in sig.parameters.items() if k not in remove_params
            )
        )
        wrapped.__signature__ = sig
        return wrapped

    return _add_logger


@add_logger()
def test_logging(log=logging.getLogger()):
    log.debug("hello logger")
    log.info("hello logger")
    log.warning("hello logger")
    log.error("hello logger")


@add_logger(show_start_end=False)
def loop_parallel(
    loop_iter: typing.Iterable,
    func: typing.Callable,
    num_cores: typing.Optional[int] = None,
    total: typing.Optional[int] = None,
    log: logging.Logger = logging.getLogger(),
    *args: typing.Any,
    **kwargs: typing.Any,
) -> list:
    """
    loop over input list and apply a function in parallel.

    Parameters
    ----------
    loop_iter : typing.Iterable
        list with inputs to `func`
    func : typing.Callable
        function to apply to each list element
    num_cores : typing.Optional[int]
        number of cores to use (Default value = None)
    total : typing.Optional[int]
        total number of iterations (Default value = None)
    log : logging.Logger (Default value = logging.getLogger())
        logger object
    args :
        position arguments passed to `func`
    kwargs :
        kwargs passed to `func`

    Returns
    -------
    type
        list with output values
    """
    if not num_cores:
        num_cores = snp2cell.NCPU
    log.info(f"using {num_cores} cores")

    inputs = tqdm(loop_iter, position=0, leave=True, total=total)

    return Parallel(n_jobs=num_cores)(delayed(func)(i, *args, **kwargs) for i in inputs)


def get_gene2pos_mapping(
    host: typing.Optional[str] = None,
    chrs: typing.Optional[list] = None,
    rev: bool = False,
) -> dict:
    """
    query biomart to get a mapping of gene symbols to genomic locations (largest range).

    Parameters
    ----------
    host : typing.Optional[str]
        biomart host, change to use archived versions (default: "http://www.ensembl.org")
    chrs : typing.Optional[list]
        chromosomes to use (default: 1-22)
    rev : bool
        reverse the dictionary and return {<genomic location>: <gene symbol>} (Default value = False)

    Returns
    -------
    type
        a dictionary {<gene symbol>: <genomic location>}
    """
    import pybiomart

    host = host or "http://www.ensembl.org"
    chrs = chrs or [str(i + 1) for i in range(22)]

    # load biomart dataset
    server = pybiomart.Server(host=host)
    mart = server["ENSEMBL_MART_ENSEMBL"]
    dataset = mart["hsapiens_gene_ensembl"]

    gene_df = dataset.query(
        attributes=[
            "chromosome_name",
            "start_position",
            "end_position",
            "strand",
            "external_gene_name",
            "ensembl_gene_id",
        ]
    )

    gene_df = gene_df.drop_duplicates("Gene name").query(
        "`Chromosome/scaffold name`.isin(@chrs)", engine="python"
    )
    gene_df["position"] = (
        "chr"
        + gene_df["Chromosome/scaffold name"].astype(str)
        + ":"
        + gene_df["Gene start (bp)"].astype(str)
        + "-"
        + gene_df["Gene end (bp)"].astype(str)
    )

    gene2pos = gene_df.set_index("Gene name")["position"].to_dict()

    if not rev:
        return gene2pos
    else:
        pos2gene = {v: k for k, v in gene2pos.items()}
        return pos2gene


def graph_nodes_to_bed(
    nx_graph: nx.Graph,
    gene2pos: typing.Optional[dict[str, str]] = None,
    chrs: typing.Optional[list[str]] = None,
    out_path: typing.Optional[str] = None,
    return_df: bool = True,
) -> typing.Optional[pd.DataFrame]:
    """
    create a BED file / return a dataframe with genomic locations for nodes in `nx_graph`

    Parameters
    ----------
    nx_graph : nx.Graph
        a networkx graph
    gene2pos : typing.Optional[dict[str, str]]
        a dict with gene symbol to genomic location mapping (default: query biomart latest version)
    chrs : typing.Optional[list[str]]
        chromosomes to use (default: 1-22)
    out_path : typing.Optional[str] (Default value = None)
        output path for BED file
    return_df : bool
        whether to return a data frame (Default value = True)

    Returns
    -------
    """
    chrs = chrs or [str(i + 1) for i in range(22)]

    if isinstance(nx_graph, str):
        with open(nx_graph, "rb") as f:
            nx_graph = pickle.load(f)
    if not gene2pos:
        gene2pos = get_gene2pos_mapping(chrs=chrs)

    network_loc_df = pd.DataFrame(index=list(nx_graph.nodes))

    network_loc_df["position"] = [
        x if re.match(".*:.*-.*", x) else None for x in network_loc_df.index.tolist()
    ]
    add_msk = network_loc_df["position"].isna() & network_loc_df.index.isin(gene2pos)
    network_loc_df["position"][add_msk] = [
        gene2pos[x] for x in network_loc_df.index[add_msk].tolist()
    ]

    network_loc_df = network_loc_df.dropna()
    network_loc_df = network_loc_df["position"].str.extract(
        "(?:chr)?(?P<Chromosome>.+):(?P<Start>.+)-(?P<End>.+)", expand=True
    )
    network_loc_df = (
        network_loc_df[network_loc_df["Chromosome"].isin(chrs)]
        .astype(int)
        .sort_values(network_loc_df.columns.tolist())
    )

    if out_path:
        network_loc_df.to_csv(out_path, sep="\t", index=False)
    if return_df:
        return network_loc_df
    else:
        return None


def filter_summ_stat(
    summ_stat_path: typing.Union[str, os.PathLike],
    network_loc: typing.Union[str, os.PathLike, pd.DataFrame],
    out_path: typing.Optional[typing.Union[str, os.PathLike]] = None,
    return_df: bool = True,
    summ_stat_kwargs: typing.Optional[dict] = None,
    network_loc_kwargs: typing.Optional[dict] = None,
) -> typing.Optional[pd.DataFrame]:
    """
    filter CSV file with summary statistics `summ_stat_path` to those overlapping locations in another file `network_loc`.
    Write results to a new file `out_path`.

    Parameters
    ----------
    summ_stat_path : typing.Union[str, os.PathLike]
        path to CSV file with summary statistics
    network_loc : typing.Union[str,os.PathLike,pd.DataFrame]
        path to CSV file with genomic locations of network nodes
    out_path : typing.Optional[typing.Union[str,os.PathLike]]
        path for output CSV file
    return_df : bool
        whether to return a pandas data frame (Default value = True)
    summ_stat_kwargs : typing.Optional[dict]
        kwargs passed to `pd.read_table(summ_stat_path, ...)`
    network_loc_kwargs : typing.Optional[dict]
        kwargs passed to `pd.read_table(network_loc, ...)`

    Returns
    -------
    type
        None or pandas `DataFrame`
    """
    import pyranges

    if network_loc_kwargs is None:
        network_loc_kwargs = {}
    if summ_stat_kwargs is None:
        summ_stat_kwargs = {}

    if isinstance(network_loc, (str, Path)):
        network_loc = pd.read_table(network_loc, **network_loc_kwargs)

    snp_df = pd.read_table(summ_stat_path, **summ_stat_kwargs)
    snp_pr = pyranges.PyRanges(snp_df)

    network_loc_pr = pyranges.PyRanges(network_loc)

    snp_pr_filt = snp_pr.overlap(network_loc_pr)

    if out_path:
        snp_pr_filt.to_csv(out_path, sep="\t")
    if return_df:
        return snp_pr_filt.df
    else:
        return None


def add_col_to_bed(
    bed: typing.Union[str, os.PathLike, pd.DataFrame],
    add_df: typing.Union[str, os.PathLike, pd.DataFrame],
    add_df_merge_on: list[str] = ["CHR", "START", "END"],
    rename: dict = {},
    drop: list = [],
    filter: typing.Optional[list[str]] = None,
    out_path: typing.Optional[typing.Union[str, os.PathLike]] = None,
    return_df: bool = True,
    bed_kwargs: dict = {},
    add_df_kwargs: dict = {},
) -> typing.Optional[pd.DataFrame]:
    """
    Merge two tables `bed` and `add_df` read from CSV files or passed as pandas `DataFrame`.
    Optionally rename columns, drop columns or filter columns to selection.
    Write resulting table to CSV and/or return pandas `DataFrame`.

    Parameters
    ----------
    bed :
        path to CSV file or pandas data frame
    add_df :
        path to CSV file or pandas data frame
    add_df_merge_on :
        columns in `add_df` specifying chromosome, start and end position
    rename :
        rename columns
    drop :
        drop columns
    filter :
        filter to selection of columns
    out_path :
        path for output dataframe as CSV file
    return_df :
        whether to return a pandas `DataFrame`
    bed_kwargs :
        kwargs passed to `pd.read_table(bed, **bed_kwargs)`
    add_df_kwargs :
        kwargs passed to `pd.read_table(add_df, **add_df_kwargs)`

    Returns
    -------
    type
        None or a pandas data frame
    """
    if isinstance(bed, (str, os.PathLike)):
        bed = pd.read_table(bed, **bed_kwargs)
    if isinstance(add_df, (str, os.PathLike)):
        add_df = pd.read_table(add_df, **add_df_kwargs)

    add_df = add_df.rename(columns=rename).drop(drop, axis=1)  # type: ignore
    add_df = add_df.filter(filter or add_df.columns.tolist())

    bed_merge_on = bed.columns.tolist()[:3]  # type: ignore
    bed = bed.astype({c: int for c in bed_merge_on})  # type: ignore
    add_df = add_df.astype({c: int for c in add_df_merge_on})

    mrg_df = bed.merge(
        add_df, how="inner", left_on=bed_merge_on, right_on=add_df_merge_on
    )
    mrg_df = mrg_df.sort_values(bed_merge_on)
    mrg_df.drop([x for x in add_df_merge_on if x not in bed_merge_on], axis=1)

    if out_path:
        mrg_df.to_csv(out_path, sep="\t", index=False)
    if return_df:
        return mrg_df
    else:
        return None


def get_reg_srt_keys(reg: str) -> tuple[int, int, int]:
    """
    from a genomic location string "(chr)?(?P<chr>.+):(?P<start>[0-9]+)-(?P<end>[0-9]+)"
    extract chromosome, start and end position (e.g. to be used for sorting).

    Parameters
    ----------
    reg :
        genomic location string

    Returns
    -------
    type
        int tuple of (<chromosome>, <start>, <end>)
    """
    m = re.match("(chr)?(?P<chr>.+):(?P<start>[0-9]+)-(?P<end>[0-9]+)", reg)
    if m:
        if re.match("[0-9]+", m.group("chr")):
            chrom_num = int(m.group("chr"))
        else:
            chrom_num = {"X": 23, "Y": 24}[m.group("chr")]
        return chrom_num, int(m.group("start")), int(m.group("end"))
    else:
        raise ValueError(f"could not match {reg}")


def get_snp_scores(
    regions: typing.Iterable[str],
    summ_stat_bed_path: typing.Union[str, os.PathLike],
    progress: bool = True,
    **kwargs: typing.Any,
) -> dict[str, float]:
    """
    read CSV file with summary statistics and compute a score for each SNP.

    Parameters
    ----------
    regions :
        list of region strings (like "chr1:234-245")
    summ_stat_bed_path :
        bath to BED file with summary statistics
    progress :
        show progress bar
    kwargs :
        kwargs for `pd.read_table(summ_stat_bed_path, **kwargs)`

    Returns
    -------
    type
        dictionary with a score for each genomic location
    """

    def get_snp_score(df):
        try:
            # snp_loc = str(df["Chromosome"]), int(df["Start"])
            snp_beta = float(df["Beta"])
            snp_se = float(df["SE"])
            snp_ld = float(df["L2"])
            if not snp_se:
                return np.nan
            snp_r = 0.1 / (0.1 + snp_se**2)
            snp_bf = (
                np.log(1 - snp_r) / 2 + (snp_beta / snp_se) ** 2 * snp_r / 2
            )  # log BF
            return np.exp(snp_bf - snp_ld/4)
        except Exception as e:
            raise ValueError(f"could not compute score using {df} \n{e}")

    summ_stat_df = pd.read_table(summ_stat_bed_path, **kwargs)
    summ_stat_pr = pyranges.PyRanges(summ_stat_df)

    reg_scores = {}

    loop_reg: typing.Union[list[str], tqdm[str]]
    if progress:
        loop_reg = tqdm(regions)
    else:
        loop_reg = list(regions)

    for reg in loop_reg:
        regm = re.match("(chr)?(?P<chr>[0-9]+):(?P<start>[0-9]+)-(?P<end>[0-9]+)", reg)
        if not regm:
            continue

        reg_chrom = str(regm.group("chr"))
        reg_start = int(regm.group("start"))
        reg_end = int(regm.group("end"))

        snp_sel = summ_stat_pr.overlap(
            pyranges.from_dict(
                {
                    "Chromosome": [reg_chrom],
                    "Start": [reg_start],
                    "End": [reg_end],
                }
            )
        ).df

        if snp_sel.shape[0] > 0:
            try:
                reg_scores[reg] = snp_sel.apply(get_snp_score, axis=1).mean()
            except Exception as e:
                print(f"could not get region score for {reg}")
                raise e

    return reg_scores


def get_snp_scores_parallel(
    regions: list[str],
    summ_stat_bed_path: typing.Union[str, os.PathLike],
    chunk_size: int = 1000,
    num_cores: typing.Optional[int] = 8,
    **kwargs: typing.Any,
) -> dict[str, float]:
    """
    read CSV file with summary statistics and compute a score for each SNP in parallel.

    Parameters
    ----------
    regions :
        list of region strings (like "chr1:234-245")
    summ_stat_bed_path :
        bath to BED file with summary statistics
    num_cores :
        number of cores to use (default: 8)
    chunk_size :
        chunk size of regions passed to each core at once
    kwargs :
        kwargs for `pd.read_table(summ_stat_bed_path, **kwargs)`

    Returns
    -------
    type
        dictionary with a score for each genomic location
    """
    region_chunks = [
        regions[i : i + chunk_size] for i in range(0, len(regions), chunk_size)
    ]

    scr_list = loop_parallel(
        region_chunks,
        get_snp_scores,
        num_cores=num_cores,
        summ_stat_bed_path=summ_stat_bed_path,
        progress=False,
        **kwargs,
    )

    reg_scores = {k: v for d in scr_list for k, v in d.items()}

    return reg_scores


def get_rank_df(
    adata: scanpy.AnnData,
    key: str = "rank_genes_groups",
    colnames: list[str] = ["names", "scores"],
) -> pd.DataFrame:
    """
    Get minimal pandas df for `rank_genes_groups` results for a specific group (also works for method='logreg').

    Parameters
    ----------
    adata :
        scanpy `AnnData` object with `rank_genes_groups` results
    key :
        key under which `rank_genes_groups` results are stored in `adata`
    colnames :
        columns to extract from `rank_genes_groups` results

    Returns
    -------
    type
        a pandas data frame
    """
    return (
        pd.concat(
            [pd.DataFrame(adata.uns[key][c]) for c in colnames],
            axis=1,
            names=[None, "group"],
            keys=colnames,
        )
        .stack(level=1)
        .reset_index()
        .sort_values(["level_0"])
        .drop(columns="level_0")
    )


def load_snp2cell(path: typing.Union[str, os.PathLike]) -> object:
    """
    Load a SNP2CELL object.

    Parameters
    ----------
    path :
        path to saved object

    Returns
    -------
    type
        SNP2CELL object
    """
    with open(path, "rb") as f:
        return dill.load(f)
