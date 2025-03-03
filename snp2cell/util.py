import logging
import os
import pickle
import re
import typing
from functools import wraps
from inspect import signature
from pathlib import Path
import multiprocessing as mp

import dill
import networkx as nx
import numpy as np
from scipy.special import logsumexp
import pandas as pd
import pyranges
import pybiomart
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

    gene_df = gene_df.drop_duplicates("external_gene_name")
    gene_df = gene_df[gene_df["chromosome_name"].isin(chrs)]
    gene_df["position"] = (
        "chr"
        + gene_df["chromosome_name"].astype(str)
        + ":"
        + gene_df["start_position"].astype(str)
        + "-"
        + gene_df["end_position"].astype(str)
    )

    gene2pos = gene_df.set_index("external_gene_name")["position"].to_dict()

    if not rev:
        return gene2pos
    else:
        pos2gene = {v: k for k, v in gene2pos.items()}
        return pos2gene


@add_logger()
def export_for_fgwas(
    s2c: "snp2cell.SNP2CELL",
    region_loc_path: str = "peak_locations.txt",
    log: logging.Logger = logging.getLogger(),
) -> None:
    """
    Export genomic regions for fgwas analysis. This can be regions in an eGRN or transcription start sites of genes.
    Transcription start sites can for example be obtained with `get_gene2pos_mapping()`.

    Assumes that nodes in the network are named like "chr1:201-701". The regions are represented by their center.

    This can be used as input files for the nf-fgwas pipeline to calculate Bayes Factors from GWAS summary statistics:
    https://github.com/cellgeni/nf-fgwas

    For this only the first part of the pipeline can be run with, e.g.:
    ```
    nextflow run /path/to/nf-fgwas/main.nf -resume -qs 1000 \
      --enrichment false \
      --window_size 5000 \
      --tss_file "/path/to/peak_locations.txt.gz" \
      --cell_types "/path/to/peak_locations.txt" \
      --studies "/path/to/studies.csv"
    ```

    Parameters
    ----------
    s2c : snp2cell.SNP2CELL
        SNP2CELL object containing the gene regulatory network (GRN)
    region_loc_path : str
        Path to save the region locations (default: "peak_locations.txt")

    Returns
    -------
    None
    """
    grn_df = nx.to_pandas_edgelist(s2c.grn)
    nodes = {
        n
        for n in grn_df["source"].tolist() + grn_df["target"].tolist()
        if n.startswith("chr")
    }

    pos_df = pd.DataFrame(
        {
            "hm_chr": [n.split(":")[0][3:] for n in nodes],
            "hm_pos": [
                (int(start) + int(end)) // 2 if start != end else int(end)
                for n in nodes
                for start, end in [n.split(":")[1].split("-")]
            ],
        }
    )

    pos_df["hm_chr"] = pd.to_numeric(pos_df["hm_chr"], errors="coerce")
    pos_df = pos_df.dropna().astype({"hm_chr": int}).sort_values(["hm_chr", "hm_pos"])

    pos_df.to_csv(f"{region_loc_path}.gz", sep="\t", header=False, index=False)
    pos_df.to_csv(region_loc_path, sep="\t", header=True, index=False)


def _calc_per_region_bf(region_id: str, region_df: pd.DataFrame) -> pd.Series:
    """
    calculate regional log Bayes factors per group (used in `load_fgwas_scores`)
    """
    log_numerator = logsumexp(region_df["SNP_BF"] + region_df["SNP_rel_loc"])
    log_denominator = logsumexp(region_df["SNP_rel_loc"])
    log_bf = log_numerator - log_denominator
    return pd.Series(log_bf, index=[region_id])


@add_logger()
def load_fgwas_scores(
    fgwas_output_path: os.PathLike,
    region_loc_path: os.PathLike,
    rbf_table_path: typing.Optional[os.PathLike] = None,
    lexpand: int = 250,
    rexpand: int = 250,
    num_cores: typing.Optional[int] = None,
    log: logging.Logger = logging.getLogger(),
) -> typing.Tuple[typing.Dict[str, float], pd.DataFrame]:
    """
    Load Bayes Factors from nf-fgwas results and calculate regional Bayes factors (RBF).

    After running the nf-fgwas pipeline (https://github.com/cellgeni/nf-fgwas), the output file should be in a folder like `results/LDSC_results/<studyid>/input.gz`.
    The region location file (nf-fgwas input) can be generated with `export_for_fgwas()`.

    The columns in the output file correspond to: region ID, SNP log BF, SNP log weight including distance and LD score.

    Returns a dictionary with RBF scores and a pandas data frame with region information (name, ID, chromosome, position, RBF).
    For the dictionary, the region locations are expanded to the full length of the region.
    This is assuming that each region was represented by its center for the fgwas analysis and that all regions have the same length (default 500bp).
    If this is not the case, you can use the returned data frame and expand the regions manually.

    Parameters
    ----------
    fgwas_output_path : os.PathLike
        Path to fgwas output file with Bayes Factors.
    region_loc_path : os.PathLike
        Path to region location file (result from `export_for_fgwas()`).
    rbf_table_path : typing.Optional[os.PathLike]
        Path to save the RBF table. Not saving table if `None`. (default: `None`)
    lexpand : int
        Number of base pairs to expand the region to the left (default: 250).
    rexpand : int
        Number of base pairs to expand the region to the right (default: 250).
    num_cores : int
        Number of cores to use (default: 8).
    log : logging.Logger
        Logger object (default: logging.getLogger()).

    Returns
    -------
    typing.Tuple[typing.Dict[str, float], pd.DataFrame]
        Dictionary with RBF scores and a pandas data frame with region
        information (name, ID, chromosome, position, RBF).
    """
    if num_cores is None:
        num_cores = snp2cell.NCPU
    log.info(f"using {num_cores} cores")

    # load fgwas output: SNP_BF (log BF), SNP_rel_loc (log weight including distance and LD score)
    df = pd.read_csv(fgwas_output_path, sep="\t", header=None)
    df.columns = ["regionID", "SNP_BF", "SNP_rel_loc"]

    with mp.Pool(num_cores) as pool:
        res = pool.starmap(_calc_per_region_bf, df.groupby("regionID"))
    res = pd.concat(res)

    # add region information from region_loc_path
    region_info = pd.read_csv(region_loc_path, sep="\t")
    region_info["log_RBF"] = region_info.index.map(res)
    region_info["name"] = region_info.apply(
        lambda r: f"chr{int(r['hm_chr'])}:{int(r['hm_pos'])}-{int(r['hm_pos'])}", axis=1
    )
    region_info["ID"] = region_info.index

    if rbf_table_path is not None:
        region_info[["name", "ID", "hm_chr", "hm_pos", "log_RBF"]].to_csv(
            rbf_table_path, sep="\t", index=False
        )

    # prepare log RBF scores
    scores = region_info.set_index("name")["log_RBF"].sort_values(ascending=False)

    def rename(s):
        # expand region to full length
        chr, pos = s.split(":")[0], int(s.split("-")[1])
        start, end = pos - lexpand, pos + rexpand
        return f"{chr}:{start}-{end}"

    scores.index = scores.index.map(rename)
    score_dct = scores.dropna().to_dict()

    return score_dct, region_info


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
            return np.exp(snp_bf - snp_ld / 4)
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
