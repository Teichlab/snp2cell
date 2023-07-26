import os
import sys
import inspect
import logging
import gc
import random
import copy
import re
import pickle
import dill
import textwrap

from pathlib import Path
import numpy as np
import scipy as sp
import pandas as pd
import statsmodels.api as sm
import networkx as nx
import scanpy as sc

import multiprocessing
from joblib import Parallel, delayed
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt


def add_logger(f):
    def wrapped(*args, **kargs):
        func_name = f.__name__

        log = logging.getLogger(func_name)
        log.setLevel(logging.DEBUG)
        if not log.hasHandlers():
            c_handler = logging.StreamHandler()
            c_handler.setLevel(logging.INFO)
            c_format = logging.Formatter(
                '[%(levelname)s - %(name)s - %(asctime)s]: %(message)s'
            )
            c_handler.setFormatter(c_format)
            log.addHandler(c_handler)

        log.info(f"----- starting {func_name} -----")
        r = f(*args, **kargs, log=log)
        log.info(f"----- finished {func_name} -----")

        return r
    return wrapped


@add_logger
def test_logging(log = logging.getLogger()):
    log.debug("hello logger")
    log.info("hello logger")
    log.warning("hello logger")
    log.error("hello logger")


@add_logger
def loop_parallel(loop_iter, func, num_cores=None, total=None, log = logging.getLogger(), *args, **kwargs):
    if not num_cores:
        num_cores = multiprocessing.cpu_count()
    log.info(f"using {num_cores} cores")
    
    inputs = tqdm(loop_iter, position=0, leave=True, total=total)

    return Parallel(n_jobs=num_cores)(
        delayed(func)(i, *args, **kwargs)
        for i in inputs
    )


def get_reg_srt_keys(reg):
    m = re.match("(chr)?(?P<chr>.+):(?P<start>[0-9]+)-(?P<end>[0-9]+)", reg)
    if re.match("[0-9]+", m.group("chr")):
        chrom_num = int(m.group("chr"))
    else:
        chrom_num = {"X": 23,"Y":24}[m.group("chr")]
    return (chrom_num, int(m.group("start")), int(m.group("end")))


def get_snp_scores(regions, summ_stat_bed_path):
    regions_srt = sorted(regions, key = get_reg_srt_keys)
    snp_bf_per_region = {}

    def read_snp(f):
        snp_info = f.readline().split("\t")
        if not snp_info or len(snp_info) < 7:
            return None, None
        snp_loc = str(snp_info[0]), int(snp_info[1])
        snp_beta = float(snp_info[5])
        snp_se = float(snp_info[6])
        snp_ld = float(snp_info[10])
        if not snp_se:
            return snp_loc, np.nan
        snp_r = 0.1 / (0.1 + snp_se**2)
        snp_bf = np.exp(np.log(1 - snp_r)/2 + (snp_beta / snp_se)**2 * snp_r / 2)
        return snp_loc, np.log(snp_bf/snp_ld)

    with open(summ_stat_bed_path, "r") as f:
        snp_loc, snp_bf = read_snp(f)

        for reg in tqdm(regions_srt):
            try:
                regm = re.match("(chr)?(?P<chr>[0-9]+):(?P<start>[0-9]+)-(?P<end>[0-9]+)", reg)
                reg_chrom = regm.group("chr")
                reg_start = regm.group("start")
                reg_end = regm.group("end")
            except:
                continue

            # print(snp_loc)
            # print(reg_chrom, reg_start, reg_end)
            
            while snp_loc and (snp_loc[0] < str(reg_chrom) or snp_loc[0] == str(reg_chrom) and snp_loc[1] < int(reg_start)):
                snp_loc, snp_bf = read_snp(f)

            all_bf = []
            while snp_loc and snp_loc[0] == str(reg_chrom) and snp_loc[1] >= int(reg_start) and snp_loc[1] <= int(reg_end):
                all_bf.append(snp_bf)
                snp_loc, snp_bf = read_snp(f)

            mean_bf = np.nanmean(all_bf)
            snp_bf_per_region[reg] = mean_bf
            
    return snp_bf_per_region


def load_snp2grn(path):
    with open(path, "rb") as f:
        return dill.load(f)


class SNP2GRN:
    def __init__(self, path=None):
        self.grn = None
        self.adata = None
        self.scores = None
        self.scores_prop = None
        self.scores_rand = {}

        if path:
            self.load_data(path)

    def __repr__(self):
        return textwrap.dedent(
            f"""
            GRN: {self.grn}
            original scores: {self.scores.shape if self.scores is not None else 'None'}
            propagated scores: {self.scores_prop.shape if self.scores_prop is not None else 'None'}
            score perturbations for: {list(self.scores_rand) if self.scores_rand else 'None'}
            anndata: {self.adata.shape if self.adata is not None else 'None'}
            """
        )

    ###--------------------------------------------------- private

    
    def _init_scores(self):
        self.scores = pd.DataFrame(index=self.grn.nodes)
        self.scores_prop = pd.DataFrame(index=self.grn.nodes)
        self.scores_rand = {}
        
    def _set_grn(self, nx_grn):
        self.grn = nx_grn.to_undirected()
        
    def _scale_score(self, score_key, which="original", inplace=True):
        std_scale = lambda s: (s - s.min()) / (s.max() - s.min())
        scores_out = {
            "propagated": self.scores_prop,
            "original": self.scores,
            "perturbed": self.scores_rand,
        }[which]
        scr = scores_out[score_key].copy()
        if which == "perturbed":
            scr = scr.apply(std_scale, axis=0)
        else:
            scr = std_scale(scr)
        scr[scr.isna()] = 0
        if inplace:
            scores_out[score_key] = scr
        else:
            return scr

    def _robust_z_score(self, series):
        mad = sp.stats.median_abs_deviation(series, scale=1.0)
        zscore = 0.6745 * (series - series.median()) / mad
        return zscore

    def _std_scale(self, series):
        return (series - series.min()) / (series.max() - series.min())
        
    def _prop_scr(self, scr_dct):
        return nx.pagerank(
            self.grn, 
            personalization = scr_dct,
        )

    def _defrag_pandas(self):
        self.scores = self.scores.copy()
        self.scores_prop = self.scores_prop.copy()
        for k in self.scores_rand.keys():
            self.scores_rand[k] = self.scores_rand[k].copy()
        gc.collect()

    ###--------------------------------------------------- input/output


    def save_obj(self, path):
        with open(path, "wb") as f:
            dill.dump(self)

    def save_data(self, path):
        data = {
            "grn": self.grn,
            "scores": self.scores,
            "scores_prop": self.scores_prop,
            "scores_rand": self.scores_rand,
            "adata": self.adata,
        }
        with open(path, "wb") as f:
            dill.dump(data, f)

    def load_data(self, path, overwrite=False):
        if self.scores and not overwrite:
            raise IndexError(
                "existing data found, set overwrite=True to discard it."
            )
        with open(path, "rb") as f:
            data = dill.load(f)
        self.grn = data["grn"]
        self.scores = data["scores"]
        self.scores_prop = data["scores_prop"]
        self.scores_rand = data["scores_rand"]
        self.adata = data["adata"]
    
    def add_grn_from_pandas(self, adjacency_df):
        pass
    
    def add_grn_from_networkx(self, nx_grn, overwrite=False):
        if self.scores and not overwrite:
            raise IndexError(
                "existing scores found, set overwrite=True to discard them."
            )
        if isinstance(nx_grn, (str, Path)):
            with open(nx_grn, "rb") as f:
                self._set_grn(pickle.load(f))
        else:
            self._set_grn(nx_grn)
        self._init_scores()
            
    def link_adata(self, adata, overwrite=False):
        if self.adata is not None:
            raise IndexError(
                "linked adata found, set overwrite=True to replace."
            )
        self.adata = adata


    ###--------------------------------------------------- scores

    @add_logger
    def add_score(self, score_dct, score_key="score", propagate=True, statistics=True, log = logging.getLogger()):
        log.info(f"adding score: {score_key}")
        self.scores[score_key] = self.scores.index.map(score_dct)
        self._scale_score(score_key)
        if propagate:
            log.info(f"propagating score: {score_key}")
            _, p_scr_dct = self.propagate_score(score_key)
            log.info(f"storing score {score_key}")
            self.scores_prop[score_key] = self.scores_prop.index.map(p_scr_dct)
        if statistics:
            self.rand_sim(score_key=score_key)
            self.add_score_statistics(score_keys=score_key)
        self._defrag_pandas()

    def propagate_score(self, score_key="score"):
        scr_dct = self.scores[score_key].to_dict()
        p_scr_dct = self._prop_scr(scr_dct)
        return (score_key, p_scr_dct)

    @add_logger
    def propagate_scores(self, score_keys, log = logging.getLogger()):
        log.info(f"propagating scores: {score_keys}")
        prop_scores = loop_parallel(score_keys, self.propagate_score)
        for key, dct in prop_scores:
            self.scores_prop[key] = self.scores_prop.index.map(dct)

    @add_logger
    def rand_sim(self, score_key="score", perturb_key=None, n=1000, log = logging.getLogger()):
        log.info(f"create {n} permutations of score {score_key}")
        if isinstance(score_key, str):
            score_key = [score_key]
        if not perturb_key:
            if len(score_key)==1:
                perturb_key = score_key[0]
            else:
                raise ValueError("must set `perturb_key` if len(score_key)>1")

        pers_rand = []
        for key in random.choices(list(score_key), k=n):
            node_list = self.scores[key].index.tolist()
            vals_list = self.scores[key].tolist()
            random.shuffle(node_list)
            
            pers_rand.append(dict(zip(node_list, vals_list)))

        log.info(f"propagating permutations")
        prop_scores = loop_parallel(pers_rand, self._prop_scr)
        log.info(f"storing scores under key {perturb_key}")
        self.scores_rand[perturb_key] = pd.DataFrame(prop_scores, index=range(len(prop_scores)))

    @add_logger
    def add_score_statistics(self, score_keys="score", log = logging.getLogger()):
        log.info(f"adding statistics for: {score_keys}")
        if isinstance(score_keys, str):
            score_keys = [score_keys]
        if isinstance(score_keys, list):
            score_keys = {k:k for k in score_keys}

        for s_key, p_key in score_keys.items():
            pval = (self.scores_prop[s_key] < self.scores_rand[p_key]).sum(axis=0) / self.scores_prop.shape[0]
            mad = sp.stats.median_abs_deviation(self.scores_rand[p_key], axis=0, scale=1.0)
            zscore = 0.6745 * (self.scores_prop[s_key] - self.scores_rand[p_key].median(axis=0)) / mad
            _, fdr, _, _ = sm.stats.multipletests(pval, method="fdr_bh")
    
            self.scores_prop[f"{s_key}__pval"] = pval
            self.scores_prop[f"{s_key}__FDR"] = fdr
            self.scores_prop[f"{s_key}__zscore"] = zscore
        self._defrag_pandas()

    @add_logger
    def combine_scores(self, combine, log = logging.getLogger()):
        """ combine: list of tuples """
        log.info(f"combining scores: {combine}")
        for key1, key2 in combine:
            scr1 = self._scale_score(score_key=key1, which="propagated", inplace=False)
            scr2 = self._scale_score(score_key=key2, which="propagated", inplace=False)
            comb_key = f"min({key1},{key2})"
            self.scores_prop[comb_key] = np.min([scr1, scr2], axis=0)

    @add_logger
    def combine_scores_rand(self, combine, log = logging.getLogger()):
        """ combine: list of tuples """
        log.info(f"combining scores: {combine}")
        for key1, key2 in combine:
            scr1 = self._scale_score(score_key=key1, which="perturbed", inplace=False)
            scr2 = self._scale_score(score_key=key2, which="perturbed", inplace=False)
            
            scr1 = scr1.loc[:,scr2.columns]
            min_n = min(scr1.shape[0], scr2.shape[0])
            scr1 = scr1.iloc[:min_n,:]
            scr2 = scr2.iloc[:min_n,:]
            
            comb_key = f"min({key1},{key2})"
            self.scores_rand[comb_key] = np.minimum(scr1, scr2)


    ###--------------------------------------------------- adata

    @add_logger
    def adata_add_de_scores(self, groupby="celltype", check=True, topn=500, simn=1000, statistics=True, log = logging.getLogger(), **kwargs):
        if check:
            if self.adata.X.max() > 50:
                raise ValueError(
                    "adata seems to contain raw counts, "
                    "lognormalise or set check=False."
                )
        if "method" not in kwargs:
            kwargs["method"] = "wilcoxon"
            kwargs["rankby_abs"] = True
        
        log.info(f"finding DE genes for annotation {groupby}")
        sc.tl.rank_genes_groups(self.adata, groupby=groupby, **kwargs)
        de_df = sc.get.rank_genes_groups_df(self.adata, group = None)
        groups = de_df.group.unique().tolist()

        score_keys = []
        for grp in groups:
            scr = de_df.query(
                f"group == '{grp}' and pvals_adj < 0.05"
            ).set_index("names")["scores"][:topn].to_dict()
            
            scr_key = f"DE_{grp}__score"
            self.add_score(scr, score_key=scr_key, propagate=False, statistics=False)
            score_keys.append(scr_key)

        self.propagate_scores(score_keys)

        if statistics:
            self.rand_sim(
                score_key = [f"DE_{grp}__score" for grp in groups],
                perturb_key = f"DE_{groupby}__score",
                n = simn,
            )
            
            self.add_score_statistics(
                score_keys = {f"DE_{grp}__score": f"DE_{groupby}__score" for grp in groups}
            )
        self._defrag_pandas()

    def adata_combine_de_scores(self, group_key="celltype", score_key="score", suffix="", statistics=True):
        groups = self.adata.obs[group_key].unique().tolist()
        
        combine = [(f"DE_{grp}__score{suffix}", score_key) for grp in groups]
        self.combine_scores(combine)
        
        if statistics:
            self.combine_scores_rand([(f"DE_{group_key}__score", score_key)])

            self.add_score_statistics(
                score_keys = {
                    f"min(DE_{grp}__score,{score_key})": f"min(DE_{group_key}__score,{score_key})" 
                    for grp in groups
                }
            )


    ###--------------------------------------------------- export

    def get_scores(self, which="propagated", query=None, sort_key=None, **kwargs):
        """
        kwargs are passed to df.filter(**kwargs)
        options for which: ['propagated', 'original', 'perturbed']
        (default: 'propagated')
        """
        scores_out = {
            "propagated": self.scores_prop,
            "original": self.scores,
            "perturbed": self.scores_rand,
        }[which]

        if which == "perturbed":
            return copy.deepcopy(scores_out)
        else:
            scores_out = scores_out.copy()

        if query:
            scores_out = scores_out.query(query)
        if sort_key:
            scores_out = scores_out.sort_values(sort_key, ascending=False)
        if kwargs:
            scores_out = scores_out.filter(**kwargs)

        return scores_out

    def remove_scores(self, which="propagated", **kwargs):
        """
        kwargs are passed to df.filter(**kwargs)
        options for which: ['propagated', 'original', 'perturbed', 'all']
        (default: 'propagated')
        """
        if which == "perturbed":
            self.scores_rand = {}
        elif which == "all":
            self._init_scores()
        elif which == "propagated":
            if kwargs:
                cols = self.scores_prop.filter(**kwargs).columns
                self.scores_prop = self.scores_prop.drop(columns=cols)
            else:
                self.scores_prop = pd.DataFrame(index=self.grn.nodes)
        elif which == "original":
            if kwargs:
                cols = self.scores.filter(**kwargs).columns
                self.scores = self.scores.drop(columns=cols)
            else:
                self.scores = pd.DataFrame(index=self.grn.nodes)

    def get_components(self, sel_nodes):
        """ return connected components """
        G_sub = self.grn.subgraph(sel_nodes)
        cc = list(nx.connected_components(G_sub))
        print([len(x) for x in cc])
        return G_sub, cc

    def get_dfs(self, sel_nodes=None, **kwargs):
        """ return pandas node and adjacency dataframes """
        adj_df = pd.DataFrame(self.grn.edges, columns=["source", "target"])
        node_df = self.get_scores(**kwargs)
        
        if sel_nodes:
            adj_df = adj_df[adj_df.source.isin(sel_nodes) & adj_df.target.isin(sel_nodes)]
            node_df = node_df[node_df.index.isin(sel_nodes)]

        #TODO: this is dataset specific, move outside of class
        node2type = {
            **{n: "TG" for n in adj_df.target[~adj_df.target.str.match(".+:[0-9]+-[0-9+]")].unique()},
            **{n: "TF" for n in adj_df.source[~adj_df.source.str.match(".+:[0-9]+-[0-9+]")].unique()},
            **{n: "region" for n in adj_df.source[adj_df.source.str.match(".+:[0-9]+-[0-9+]")].unique()},
            **{n: "region" for n in adj_df.target[adj_df.target.str.match(".+:[0-9]+-[0-9+]")].unique()},
        }
        
        adj_df["interaction_type"] = [f"{node2type[r['source']]}_{node2type[r['target']]}" for _,r in adj_df.iterrows()]
        node_df["node_type"] = node_df.index.map(node2type)

        return node_df, adj_df

    def plot_group_summary(self, std_cutoff=1, row_pattern=".*DE_(?P<rowname>.+?)__", **kwargs):
        """ extract columns using like='' or regex='' and plot boxplot """
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff,:]
        # plt_df = plt_df.loc[~plt_df.apply(lambda x: all(x[0]==x), axis=1),:]
        plt_df = plt_df.apply(self._robust_z_score, axis=1)        
        
        plt_df = plt_df.rename(columns=lambda c: re.match(row_pattern, c).group("rowname"))
        
        sns.set(style="ticks")
        sns.set_style("whitegrid")
        sns.boxplot(
            data = plt_df, 
            order = plt_df.median(axis=0).sort_values(ascending=False).index.tolist(), 
            showfliers = False, 
            notch = True, 
            boxprops = {'facecolor':'#3C5488FF', 'edgecolor':'none'}, 
            medianprops = {'color':'lightblue'},
            width = 0.7,
        )
        plt.xticks(rotation=60, horizontalalignment="right")

    def plot_group_heatmap(self, std_cutoff=1, row_pattern=".*DE_(?P<rowname>.+?)__", **kwargs):
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff,:]
        plt_df = plt_df.apply(self._std_scale, axis=1)        
        
        plt_df = plt_df.rename(columns=lambda c: re.match(row_pattern, c).group("rowname"))
        
        plt_df = plt_df.loc[:,plt_df.median(axis=0).sort_values(ascending=False).index.tolist()]
        rows = []
        for c in plt_df:
            rows.extend(plt_df[plt_df[c]==1].index.tolist())
        plt_df = plt_df.loc[rows,:]
        
        sns.heatmap(plt_df, cmap="mako", yticklabels=False)
