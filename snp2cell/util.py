import logging
import pickle
import re
from functools import wraps
from inspect import signature
from pathlib import Path

import dill
import numpy as np
import pandas as pd
import pyranges
from joblib import Parallel, delayed
from tqdm import tqdm

import snp2cell


def set_num_cpu(n):
    snp2cell.NCPU = n


def add_logger(show_start_end=True):
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
    loop_iter,
    func,
    num_cores=None,
    total=None,
    log=logging.getLogger(),
    *args,
    **kwargs,
):
    if not num_cores:
        num_cores = NCPU
    log.info(f"using {num_cores} cores")

    inputs = tqdm(loop_iter, position=0, leave=True, total=total)

    return Parallel(n_jobs=num_cores)(delayed(func)(i, *args, **kwargs) for i in inputs)


def get_gene2pos_mapping(host=None, chrs=None, rev=False):
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
    nx_graph, gene2pos=None, chrs=None, out_path=None, return_df=True
):
    chrs = chrs or [str(i + 1) for i in range(22)]

    if isinstance(nx_graph, str):
        with open(nx_graph, "rb") as f:
            nx_graph = pickle.load(f)
    if not gene2pos:
        gene2pos = get_gene2pos_mapping(chrs=chrs)

    network_loc_df = pd.DataFrame(index=nx_graph.nodes)

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


def filter_summ_stat(
    summ_stat_path,
    network_loc,
    out_path=None,
    return_df=True,
    summ_stat_kwargs=None,
    network_loc_kwargs=None,
):
    """
    @param summ_stat_kwargs:
    @param network_loc_kwargs:
    @param return_df:
    @param out_path:
    @param network_loc:
    @param summ_stat_path:
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
        snp_pr_filt.to_csv(out_path, sep="\t", index=False)
    if return_df:
        return snp_pr_filt.df


def add_col_to_bed(
    bed,
    add_df,
    add_df_merge_on=["CHR", "START", "END"],
    rename={},
    drop=[],
    filter=None,
    out_path=None,
    return_df=True,
    bed_kwargs={},
    add_df_kwargs={},
):
    if isinstance(bed, (str, Path)):
        bed = pd.read_table(bed, **bed_kwargs)
    if isinstance(add_df, (str, Path)):
        add_df = pd.read_table(add_df, **add_df_kwargs)

    add_df = add_df.rename(columns=rename).drop(drop, axis=1)
    add_df = add_df.filter(filter or add_df.columns.tolist())

    bed_merge_on = bed.columns.tolist()[:3]
    bed = bed.astype({c: int for c in bed_merge_on})
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


def get_reg_srt_keys(reg):
    m = re.match("(chr)?(?P<chr>.+):(?P<start>[0-9]+)-(?P<end>[0-9]+)", reg)
    if re.match("[0-9]+", m.group("chr")):
        chrom_num = int(m.group("chr"))
    else:
        chrom_num = {"X": 23, "Y": 24}[m.group("chr")]
    return chrom_num, int(m.group("start")), int(m.group("end"))


def get_snp_scores(regions, summ_stat_bed_path, progress=True, **kwargs):
    def get_snp_score(df):
        try:
            # snp_loc = str(df["Chromosome"]), int(df["Start"])
            snp_beta = float(df["Beta"])
            snp_se = float(df["SE"])
            snp_ld = float(df["L2"])
            if not snp_se:
                return np.nan
            snp_r = 0.1 / (0.1 + snp_se**2)
            snp_bf = np.exp(
                np.log(1 - snp_r) / 2 + (snp_beta / snp_se) ** 2 * snp_r / 2
            )
            return np.log(snp_bf) / snp_ld
        except Exception as e:
            raise ValueError(f"could not compute score using {df} \n{e}")

    summ_stat_df = pd.read_table(summ_stat_bed_path, **kwargs)
    summ_stat_pr = pyranges.PyRanges(summ_stat_df)

    reg_scores = {}

    if progress:
        loop_reg = tqdm(regions)
    else:
        loop_reg = list(regions)

    for reg in loop_reg:
        try:
            regm = re.match(
                "(chr)?(?P<chr>[0-9]+):(?P<start>[0-9]+)-(?P<end>[0-9]+)", reg
            )
            reg_chrom = str(regm.group("chr"))
            reg_start = int(regm.group("start"))
            reg_end = int(regm.group("end"))
        except:
            continue

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
    regions, summ_stat_bed_path, chunk_size=1000, num_cores=8, **kwargs
):
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


def get_rank_df(adata, key="rank_genes_groups", colnames=["names", "scores"]):
    """get minimal df for a specific group (also works for method='logreg')"""
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


def load_snp2cell(path):
    with open(path, "rb") as f:
        return dill.load(f)
