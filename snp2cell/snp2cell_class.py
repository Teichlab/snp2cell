import copy
import gc
import logging
import multiprocessing
import pickle
import random
import re
import textwrap
from pathlib import Path

import dill
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
import scipy as sp
import seaborn as sns
import statsmodels.api as sm
from snp2cell.util import add_logger, loop_parallel, get_rank_df

NCPU = multiprocessing.cpu_count()


class SNP2CELL:
    def __init__(self, path=None):
        self.grn = None
        self.adata = None
        self.scores = None
        self.scores_prop = None
        self.scores_rand = {}
        self.de_groups = {}

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

    def _add_de_groups(self, groupby, groups):
        if groupby in self.de_groups:
            raise ValueError(f"group key {groupby} already exists")

        for k, v in self.de_groups.items():
            shared_keys = set(groups) & set(v)
            if len(shared_keys) > 0:
                raise ValueError(
                    f"groups {shared_keys} already exists under key {groupby}"
                )
        self.de_groups[groupby] = groups

    def _scale_score(self, score_key=None, score=None, which="original", inplace=True):
        assert score_key is not None or score is not None
        scores_out = {
            "propagated": self.scores_prop,
            "original": self.scores,
            "perturbed": self.scores_rand,
        }[which]
        if score_key:
            scr = scores_out[score_key].copy()
        else:
            scr = score.copy()
        if which == "perturbed":
            scr = scr.apply(self._std_scale, axis=1)
        else:
            scr = self._std_scale(scr)
        scr[scr.isna()] = 0
        if inplace:
            scores_out[score_key] = scr
        else:
            return scr

    def _get_perturbed_stats(self, score_key, suffix):
        # TODO: implement more suffixes
        assert suffix in ["__zscore"]
        score = self.scores_rand[score_key]
        if suffix == "__zscore":
            score = score.apply(self._z_score, axis=0)
        return score

    def _robust_z_score(self, series):
        mad = sp.stats.median_abs_deviation(series, scale=1.0)
        zscore = 0.6745 * (series - series.median()) / mad
        return zscore

    def _z_score(self, series):
        return (series - series.mean()) / series.std()

    def _std_scale(self, series):
        return (series - series.min()) / (series.max() - series.min())

    def _prop_scr(self, scr_dct):
        return nx.pagerank(
            self.grn,
            personalization=scr_dct,
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
            dill.dump(self, f)

    def save_data(self, path):
        data = {
            "grn": self.grn,
            "scores": self.scores,
            "scores_prop": self.scores_prop,
            "scores_rand": self.scores_rand,
            "adata": self.adata,
            "de_groups": self.de_groups,
        }
        with open(path, "wb") as f:
            dill.dump(data, f)

    def load_data(self, path, overwrite=False):
        if self.scores and not overwrite:
            raise IndexError("existing data found, set overwrite=True to discard it.")
        with open(path, "rb") as f:
            data = dill.load(f)
        self.grn = data["grn"]
        self.scores = data["scores"]
        self.scores_prop = data["scores_prop"]
        self.scores_rand = data["scores_rand"]
        self.adata = data["adata"]
        self.de_groups = data["de_groups"]

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
            raise IndexError("linked adata found, set overwrite=True to replace.")
        self.adata = adata

    ###--------------------------------------------------- scores

    @add_logger(show_start_end=False)
    def add_score(
        self,
        score_dct,
        score_key="score",
        propagate=True,
        statistics=True,
        num_cores=None,
        log=logging.getLogger(),
    ):
        log.info(f"adding score: {score_key}")
        if score_key in self.scores:
            log.warning(f"overwriting existing score self.scores['{score_key}']")
        self.scores[score_key] = self.scores.index.map(score_dct)
        self._scale_score(score_key)
        if propagate:
            log.info(f"propagating score: {score_key}")
            _, p_scr_dct = self.propagate_score(score_key)
            log.info(f"storing score {score_key}")
            if score_key in self.scores_prop:
                log.warning(
                    f"overwriting existing score self.scores_prop['{score_key}']"
                )
            self.scores_prop[score_key] = self.scores_prop.index.map(p_scr_dct)
        if statistics:
            self.rand_sim(score_key=score_key, num_cores=num_cores)
            self.add_score_statistics(score_keys=score_key)
        self._defrag_pandas()

    def propagate_score(self, score_key="score"):
        scr_dct = self.scores[score_key].to_dict()
        p_scr_dct = self._prop_scr(scr_dct)
        return (score_key, p_scr_dct)

    @add_logger()
    def propagate_scores(self, score_keys, num_cores=None, log=logging.getLogger()):
        log.info(f"propagating scores: {score_keys}")
        prop_scores = loop_parallel(
            score_keys, self.propagate_score, num_cores=num_cores
        )
        for key, dct in prop_scores:
            if key in self.scores_prop:
                log.warning(f"overwriting existing score self.scores_prop['{key}']")
            self.scores_prop[key] = self.scores_prop.index.map(dct)

    @add_logger()
    def rand_sim(
        self,
        score_key="score",
        perturb_key=None,
        n=1000,
        num_cores=None,
        log=logging.getLogger(),
    ):
        log.info(f"create {n} permutations of score {score_key}")
        if isinstance(score_key, str):
            score_key = [score_key]
        if not perturb_key:
            if len(score_key) == 1:
                perturb_key = score_key[0]
            else:
                raise ValueError("must set `perturb_key` if len(score_key)>1")

        pers_rand = []
        for key in random.choices(list(score_key), k=n):
            node_list = self.scores[key].index.tolist()
            vals_list = self.scores[key].tolist()
            random.shuffle(node_list)

            pers_rand.append(dict(zip(node_list, vals_list)))

        log.info("propagating permutations")
        prop_scores = loop_parallel(pers_rand, self._prop_scr, num_cores=num_cores)
        log.info(f"storing scores under key {perturb_key}")
        self.scores_rand[perturb_key] = pd.DataFrame(
            prop_scores, index=range(len(prop_scores))
        )

    @add_logger()
    def add_score_statistics(self, score_keys="score", log=logging.getLogger()):
        log.info(f"adding statistics for: {score_keys}")
        if isinstance(score_keys, str):
            score_keys = [score_keys]
        if isinstance(score_keys, list):
            score_keys = {k: k for k in score_keys}

        dfs = []
        for s_key, p_key in score_keys.items():
            pval = (self.scores_prop[s_key] < self.scores_rand[p_key]).sum(
                axis=0
            ) / self.scores_rand[p_key].shape[0]
            _, fdr, _, _ = sm.stats.multipletests(pval, method="fdr_bh")
            zscore_std = (
                self.scores_prop[s_key] - self.scores_rand[p_key].mean(axis=0)
            ) / self.scores_rand[p_key].std(axis=0)
            mad = sp.stats.median_abs_deviation(
                self.scores_rand[p_key], axis=0, scale=1.0
            )
            zscore_mad = (
                0.6745
                * (self.scores_prop[s_key] - self.scores_rand[p_key].median(axis=0))
                / mad
            )

            dfs.append(
                pd.DataFrame(
                    {
                        f"{s_key}__pval": pval,
                        f"{s_key}__FDR": fdr,
                        f"{s_key}__zscore": zscore_std,
                        f"{s_key}__zscore_mad": zscore_mad,
                    },
                    index=self.scores_prop.index,
                )
            )

        new_cols = set([c for df in dfs for c in df.columns.tolist()])
        old_cols = set(self.scores_prop.columns.tolist())
        existing_cols = new_cols & old_cols

        if len(existing_cols) > 0:
            log.warning(f"overwriting existing score statistics: {existing_cols}")
            self.scores_prop = self.scores_prop.drop(list(existing_cols), axis=1)

        self.scores_prop = pd.concat([self.scores_prop] + dfs, axis=1)
        self._defrag_pandas()

    @add_logger()
    def combine_scores(self, combine, scale=False, log=logging.getLogger()):
        """combine: list of tuples"""
        log.info(f"combining scores: {combine}")
        for key1, key2 in combine:
            if scale:
                scr1 = self._scale_score(
                    score_key=key1, which="propagated", inplace=False
                )
                scr2 = self._scale_score(
                    score_key=key2, which="propagated", inplace=False
                )
            else:
                scr1 = self.scores_prop[key1]
                scr2 = self.scores_prop[key2]
            comb_key = f"min({key1},{key2})"
            self.scores_prop[comb_key] = np.min([scr1, scr2], axis=0)

    @add_logger()
    def combine_scores_rand(self, combine, scale=False, log=logging.getLogger()):
        """combine: list of tuples"""
        log.info(f"combining scores: {combine}")
        for key1, key2, suffix in combine:
            if not suffix:
                if scale:
                    scr1 = self._scale_score(
                        score_key=key1, which="perturbed", inplace=False
                    )
                    scr2 = self._scale_score(
                        score_key=key2, which="perturbed", inplace=False
                    )
                else:
                    scr1 = self.scores_rand[key1]
                    scr2 = self.scores_rand[key2]
            else:
                scr1 = self._get_perturbed_stats(score_key=key1, suffix=suffix)
                scr2 = self._get_perturbed_stats(score_key=key2, suffix=suffix)
                if scale:
                    scr1 = self._scale_score(
                        score=scr1, which="perturbed", inplace=False
                    )
                    scr2 = self._scale_score(
                        score=scr2, which="perturbed", inplace=False
                    )

            scr1 = scr1.loc[:, scr2.columns]
            min_n = min(scr1.shape[0], scr2.shape[0])
            scr1 = scr1.iloc[:min_n, :]
            scr2 = scr2.iloc[:min_n, :]

            comb_key = f"min({key1}{suffix},{key2}{suffix})"
            self.scores_rand[comb_key] = np.minimum(scr1, scr2)

    ###--------------------------------------------------- adata

    @add_logger()
    def adata_add_de_scores(
        self,
        groupby="celltype",
        check=True,
        topn=500,
        simn=1000,
        statistics=True,
        num_cores=None,
        log=logging.getLogger(),
        rank_by="up",
        **kwargs,
    ):
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

        if "method" in kwargs and kwargs["method"] == "logreg":
            de_df = get_rank_df(self.adata)
        else:
            de_df = sc.get.rank_genes_groups_df(self.adata, group=None)

        if rank_by == "abs":
            log.info("ranking by up- and downregulation...")
            de_df["scores"] = de_df["scores"].abs()
        elif rank_by == "down":
            log.info("ranking by downregulation...")
            de_df["scores"] = de_df["scores"] * -1
        else:
            log.info("ranking by upregulation...")

        de_df = de_df.sort_values("scores", ascending=False)
        groups = de_df.group.unique().tolist()
        self._add_de_groups(groupby, groups)

        score_keys = []
        for grp in groups:
            if "method" in kwargs and kwargs["method"] == "logreg":
                query_str = f"group == '{grp}'"
            else:
                query_str = f"group == '{grp}' and pvals_adj < 0.05"
            scr = de_df.query(query_str).set_index("names")["scores"][:topn].to_dict()

            scr_key = f"DE_{grp}__score"
            self.add_score(scr, score_key=scr_key, propagate=False, statistics=False)
            score_keys.append(scr_key)

        self.propagate_scores(score_keys, num_cores=num_cores)

        if statistics:
            self.rand_sim(
                score_key=[f"DE_{grp}__score" for grp in groups],
                perturb_key=f"DE_{groupby}__score",
                n=simn,
                num_cores=num_cores,
            )

            self.add_score_statistics(
                score_keys={
                    f"DE_{grp}__score": f"DE_{groupby}__score" for grp in groups
                }
            )
        self._defrag_pandas()

    def adata_combine_de_scores(
        self, group_key="celltype", score_key="score", suffix="", statistics=True
    ):
        groups = self.de_groups[group_key]

        combine = [
            (f"DE_{grp}__score{suffix}", f"{score_key}{suffix}") for grp in groups
        ]
        self.combine_scores(combine)

        if statistics:
            self.combine_scores_rand([(f"DE_{group_key}__score", score_key, suffix)])

            self.add_score_statistics(
                score_keys={
                    f"min(DE_{grp}__score{suffix},{score_key}{suffix})": f"min(DE_{group_key}__score{suffix},{score_key}{suffix})"
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
        """return connected components"""
        G_sub = self.grn.subgraph(sel_nodes)
        cc = list(nx.connected_components(G_sub))
        print([len(x) for x in cc])
        return G_sub, cc

    def get_dfs(self, sel_nodes=None, **kwargs):
        """return pandas node and adjacency dataframes"""
        adj_df = pd.DataFrame(self.grn.edges, columns=["source", "target"])
        node_df = self.get_scores(**kwargs)

        if sel_nodes:
            adj_df = adj_df[
                adj_df.source.isin(sel_nodes) & adj_df.target.isin(sel_nodes)
            ]
            node_df = node_df[node_df.index.isin(sel_nodes)]

        # TODO: this is dataset specific, move outside of class
        node2type = {
            **{
                n: "TG"
                for n in adj_df.target[
                    ~adj_df.target.str.match(".+:[0-9]+-[0-9+]")
                ].unique()
            },
            **{
                n: "TF"
                for n in adj_df.source[
                    ~adj_df.source.str.match(".+:[0-9]+-[0-9+]")
                ].unique()
            },
            **{
                n: "region"
                for n in adj_df.source[
                    adj_df.source.str.match(".+:[0-9]+-[0-9+]")
                ].unique()
            },
            **{
                n: "region"
                for n in adj_df.target[
                    adj_df.target.str.match(".+:[0-9]+-[0-9+]")
                ].unique()
            },
        }

        adj_df["interaction_type"] = [
            f"{node2type[r['source']]}_{node2type[r['target']]}"
            for _, r in adj_df.iterrows()
        ]
        node_df["node_type"] = node_df.index.map(node2type)

        return node_df, adj_df

    def plot_group_summary(
        self, std_cutoff=1, row_pattern=".*DE_(?P<rowname>.+?)__", **kwargs
    ):
        """extract columns using like='' or regex='' and plot boxplot"""
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff, :]
        # plt_df = plt_df.loc[~plt_df.apply(lambda x: all(x[0]==x), axis=1),:]
        plt_df = plt_df.apply(self._robust_z_score, axis=1)

        plt_df = plt_df.rename(
            columns=lambda c: re.match(row_pattern, c).group("rowname")
        )

        sns.set(style="ticks")
        sns.set_style("whitegrid")
        sns.boxplot(
            data=plt_df,
            order=plt_df.median(axis=0).sort_values(ascending=False).index.tolist(),
            showfliers=False,
            notch=True,
            boxprops={"facecolor": "#3C5488FF", "edgecolor": "none"},
            medianprops={"color": "lightblue"},
            width=0.7,
        )
        plt.xticks(rotation=60, horizontalalignment="right")

    def plot_group_heatmap(
        self, std_cutoff=1, row_pattern=".*DE_(?P<rowname>.+?)__", **kwargs
    ):
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff, :]
        plt_df = plt_df.apply(self._std_scale, axis=1)

        plt_df = plt_df.rename(
            columns=lambda c: re.match(row_pattern, c).group("rowname")
        )

        plt_df = plt_df.loc[
            :, plt_df.median(axis=0).sort_values(ascending=False).index.tolist()
        ]
        rows = []
        for c in plt_df:
            rows.extend(plt_df[plt_df[c] == 1].index.tolist())
        plt_df = plt_df.loc[rows, :]

        sns.heatmap(plt_df, cmap="mako", yticklabels=False)
