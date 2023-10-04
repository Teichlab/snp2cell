import copy
import gc
import logging
import multiprocessing
import os
import pickle
import random
import re
import textwrap
from pathlib import Path
from typing import Union, Optional, Any, Iterable

import dill
import matplotlib.pyplot as plt  # type: ignore
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
    def __init__(self, path: Optional[Union[str, os.PathLike]] = None) -> None:
        # TODO: dynamic change from None to other values requires too much "type: ignore"
        self.grn: Optional[nx.Graph] = None
        self.adata: Optional[sc.AnnData] = None
        self.scores: Optional[pd.DataFrame] = None
        self.scores_prop: Optional[pd.DataFrame] = None
        self.scores_rand: dict[str, pd.DataFrame] = {}
        self.de_groups: dict[str, list[str]] = {}

        if path:
            self.load_data(path)

    def __repr__(self) -> str:
        return textwrap.dedent(
            f"""
            GRN: {self.grn}
            original scores: {self.scores.shape if self.scores is not None else 'None'}  # type: ignore
            propagated scores: {self.scores_prop.shape if self.scores_prop is not None else 'None'}  # type: ignore
            score perturbations for: {list(self.scores_rand) if self.scores_rand else 'None'}
            anndata: {self.adata.shape if self.adata is not None else 'None'}  # type: ignore
            """
        )

    ###--------------------------------------------------- private

    def _init_scores(self) -> None:
        if self.grn:
            self.scores = pd.DataFrame(index=list(self.grn.nodes))
            self.scores_prop = pd.DataFrame(index=list(self.grn.nodes))
            self.scores_rand = {}
        else:
            raise ValueError("No GRN set, add GRN first.")

    def _set_grn(self, nx_grn: nx.Graph) -> None:
        self.grn = nx_grn.to_undirected()

    def _add_de_groups(self, groupby: str, groups: list[str]) -> None:
        if groupby in self.de_groups:
            raise ValueError(f"group key {groupby} already exists")

        for k, v in self.de_groups.items():
            shared_keys = set(groups) & set(v)
            if len(shared_keys) > 0:
                raise ValueError(
                    f"groups {shared_keys} already exists under key {groupby}"
                )
        self.de_groups[groupby] = groups

    def _scale_score(
        self,
        score_key: Optional[str] = None,
        score: Optional[Union[pd.Series, pd.DataFrame]] = None,
        which: str = "original",
        inplace: bool = True,
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        assert score_key is not None or score is not None
        scores_out: Optional[Union[pd.DataFrame, dict[str, pd.DataFrame]]]
        scores_out = {
            "propagated": self.scores_prop,
            "original": self.scores,
            "perturbed": self.scores_rand,
        }[which]
        if not scores_out:
            raise ValueError("Initialise scores first.")
        scr: Union[pd.DataFrame, pd.Series]
        if score_key:
            scr = scores_out[score_key].copy()
        else:
            scr = score.copy()  # type: ignore
        if which == "perturbed":
            scr = scr.apply(self._std_scale, axis=1)
        else:
            scr = self._std_scale(scr)  # type: ignore
        scr[scr.isna()] = 0
        if inplace:
            if not score_key:
                raise ValueError("Need `score_key` for inplace assignment.")
            scores_out[score_key] = scr  # type: ignore
            return None
        else:
            return scr

    def _get_perturbed_stats(self, score_key: str, suffix: str) -> pd.DataFrame:
        # TODO: implement more suffixes
        assert suffix in ["__zscore"]
        score = self.scores_rand[score_key]
        if suffix == "__zscore":
            score = score.apply(self._z_score, axis=0)
        return score

    @staticmethod
    def _robust_z_score(series: pd.Series) -> pd.Series:
        mad = sp.stats.median_abs_deviation(series, scale=1.0)
        zscore = 0.6745 * (series - series.median()) / mad
        return zscore

    @staticmethod
    def _z_score(series: pd.Series) -> pd.Series:
        return (series - series.mean()) / series.std()

    @staticmethod
    def _std_scale(series: pd.Series) -> pd.Series:
        return (series - series.min()) / (series.max() - series.min())

    def _prop_scr(self, scr_dct: dict[str, float]) -> dict[str, float]:
        return nx.pagerank(
            self.grn,
            personalization=scr_dct,
        )

    def _defrag_pandas(self) -> None:
        assert self.scores and self.scores_prop and self.scores_rand
        self.scores = self.scores.copy()
        self.scores_prop = self.scores_prop.copy()
        for k in self.scores_rand.keys():
            self.scores_rand[k] = self.scores_rand[k].copy()
        gc.collect()

    def _check_init(self, check_adata: bool = False) -> None:
        if not self.scores or not self.scores_prop or not self.scores_rand:
            raise ValueError("Need to init scores first.")
        if not self.grn:
            raise ValueError("Need to set GRN first.")
        if check_adata and not self.adata:
            raise ValueError("Need to add AnnData first")

    ###--------------------------------------------------- input/output

    def save_obj(self, path: Union[str, os.PathLike]) -> None:
        """deprecated"""
        with open(path, "wb") as f:
            dill.dump(self, f)

    def save_data(self, path: Union[str, os.PathLike]) -> None:
        """
        Save object data. Reload using `s2c = SNP2CELL(<path>)`.
        :param path: output path for pickle file
        :return: None
        """
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

    def load_data(self, path: Union[str, os.PathLike], overwrite: bool = False) -> None:
        """
        Load data into object.

        :param path: input path for pickle file with data
        :param overwrite: whether to overwrite existing data
        :return: None
        """
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
        """not implemented"""
        pass

    def add_grn_from_networkx(self, nx_grn: nx.Graph, overwrite: bool = False) -> None:
        """
        Add GRN from networkx object to snp2cell object.
        :param nx_grn: networkx object
        :param overwrite: whether to overwrite existing networkx object
        :return: None
        """
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

    def link_adata(self, adata: sc.AnnData, overwrite: bool = False) -> None:
        """
        Add scanpy AnnData to snp2cell object.

        :param adata: scnapy AnnData object
        :param overwrite: whether to overwrite existing AnnData object
        :return: None
        """
        if self.adata is not None and not overwrite:
            raise IndexError("linked adata found, set overwrite=True to replace.")
        self.adata = adata

    ###--------------------------------------------------- scores

    @add_logger(show_start_end=False)
    def add_score(
        self,
        score_dct: dict[str, float],
        score_key: str = "score",
        propagate: bool = True,
        statistics: bool = True,
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Add a score to the object. Optionally propagate the score and calculate permutation statistics.

        :param score_dct: dictionary with score values
        :param score_key: key under which to store the scores
        :param propagate: whether to propagate the scores
        :param statistics: whether to calculate statistics
        :param num_cores: number of cores to use
        :param log: logger
        :return: None
        """
        self._check_init()
        log.info(f"adding score: {score_key}")
        if score_key in self.scores:  # type: ignore
            log.warning(f"overwriting existing score self.scores['{score_key}']")
        self.scores[score_key] = self.scores.index.map(score_dct)  # type: ignore
        self._scale_score(score_key)
        if propagate:
            log.info(f"propagating score: {score_key}")
            _, p_scr_dct = self.propagate_score(score_key)
            log.info(f"storing score {score_key}")
            if score_key in self.scores_prop:  # type: ignore
                log.warning(
                    f"overwriting existing score self.scores_prop['{score_key}']"
                )
            self.scores_prop[score_key] = self.scores_prop.index.map(p_scr_dct)  # type: ignore
        if statistics:
            self.rand_sim(score_key=score_key, num_cores=num_cores)
            self.add_score_statistics(score_keys=score_key)
        self._defrag_pandas()

    def propagate_score(self, score_key: str = "score") -> tuple[str, dict[str, float]]:
        """
        Propagate score.
        :param score_key: key under which the score is stored.
        :return: tuple of the score key and dict with score values after propagation
        """
        self._check_init()
        scr_dct = self.scores[score_key].to_dict()  # type: ignore
        p_scr_dct = self._prop_scr(scr_dct)
        return score_key, p_scr_dct

    @add_logger()
    def propagate_scores(
        self,
        score_keys: list[str],
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Propagate multiple scores.
        :param score_keys: keys under which the scores are stored
        :param num_cores: number of cores to use
        :param log: logger
        :return: None
        """
        self._check_init()
        log.info(f"propagating scores: {score_keys}")
        prop_scores = loop_parallel(
            score_keys, self.propagate_score, num_cores=num_cores
        )
        for key, dct in prop_scores:  # type: ignore
            if key in self.scores_prop:  # type: ignore
                log.warning(f"overwriting existing score self.scores_prop['{key}']")
            self.scores_prop[key] = self.scores_prop.index.map(dct)  # type: ignore

    @add_logger()
    def rand_sim(
        self,
        score_key: Union[str, list[str]] = "score",
        perturb_key: Optional[str] = None,
        n: int = 1000,
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Create random permutations and propagate them.
        :param score_key: key of score for which to compute statistics
        :param perturb_key: key under which to store permutation results
        :param n: number of permutation to create (default: 1000)
        :param num_cores: number of cores to use
        :param log: logger
        :return: None
        """
        self._check_init()
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
            node_list = self.scores[key].index.tolist()  # type: ignore
            vals_list = self.scores[key].tolist()  # type: ignore
            random.shuffle(node_list)

            pers_rand.append(dict(zip(node_list, vals_list)))

        log.info("propagating permutations")
        prop_scores = loop_parallel(pers_rand, self._prop_scr, num_cores=num_cores)
        log.info(f"storing scores under key {perturb_key}")
        self.scores_rand[perturb_key] = pd.DataFrame(
            prop_scores, index=range(len(prop_scores))
        )

    @add_logger()
    def add_score_statistics(
        self,
        score_keys: Union[str, list[str], dict[str, str]] = "score",
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Calculate statistics from random permutations for scores
        :param score_keys: scores for which to calculate statistics. May be a single score, a list of scores or a dictionary ({<score_key>: <perturb_key>}). If not a dictionary assuming `score_key==perturb_key`.
        :param log: logger
        :return: None
        """
        self._check_init()
        log.info(f"adding statistics for: {score_keys}")
        if isinstance(score_keys, str):
            score_keys = [score_keys]
        if isinstance(score_keys, list):
            score_keys = {k: k for k in score_keys}

        dfs = []
        for s_key, p_key in score_keys.items():
            pval = (self.scores_prop[s_key] < self.scores_rand[p_key]).sum(  # type: ignore
                axis=0
            ) / self.scores_rand[
                p_key
            ].shape[
                0
            ]
            _, fdr, _, _ = sm.stats.multipletests(pval, method="fdr_bh")
            zscore_std = (
                self.scores_prop[s_key] - self.scores_rand[p_key].mean(axis=0)  # type: ignore
            ) / self.scores_rand[p_key].std(axis=0)
            mad = sp.stats.median_abs_deviation(
                self.scores_rand[p_key], axis=0, scale=1.0
            )
            zscore_mad = (
                0.6745
                * (self.scores_prop[s_key] - self.scores_rand[p_key].median(axis=0))  # type: ignore
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
                    index=self.scores_prop.index,  # type: ignore
                )
            )

        new_cols = set([c for df in dfs for c in df.columns.tolist()])
        old_cols = set(self.scores_prop.columns.tolist())  # type: ignore
        existing_cols = new_cols & old_cols

        if len(existing_cols) > 0:
            log.warning(f"overwriting existing score statistics: {existing_cols}")
            self.scores_prop = self.scores_prop.drop(list(existing_cols), axis=1)  # type: ignore

        concat_dfs: list[pd.DataFrame] = [self.scores_prop] + dfs  # type: ignore
        self.scores_prop = pd.concat(concat_dfs, axis=1)
        self._defrag_pandas()

    @add_logger()
    def combine_scores(
        self,
        combine: Iterable[tuple[str, str]],
        scale: bool = False,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Combine pairs of scores. (E.g. for combining scores from GWAS data and differential expression).
        Currently, the only implemented option for combining the scores is min(score1, score2).
        New scores will be added, called `min(<key1>,<key2>)` for each score pair.

        :param combine: a list of pairs of scores to combine
        :param scale: whether to scale scores between 0 and 1 before combining
        :param log: logger
        :return: None
        """
        self._check_init()
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
                scr1 = self.scores_prop[key1]  # type: ignore
                scr2 = self.scores_prop[key2]  # type: ignore
            comb_key = f"min({key1},{key2})"
            self.scores_prop[comb_key] = np.min([scr1, scr2], axis=0)  # type: ignore

    @add_logger()
    def combine_scores_rand(
        self,
        combine: Iterable[tuple[str, str, str]],
        scale: bool = False,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Create random permutation background for combined scores.
        Currently, the only implemented option for combining the scores is min(score1, score2).

        :param combine: list of score pairs for which to create random permutation background. This is a list of triples, also containing an optional suffix for each pair. Currently, the only allowed suffix is "__zscore" (which will combine zscores instead of using permuted scores directly).
        :param scale: whether to scale scores between 0 and 1 before combining
        :param log: logger
        :return: None
        """
        log.info(f"combining scores: {combine}")
        for key1, key2, suffix in combine:
            scr1: pd.DataFrame
            scr2: pd.DataFrame
            if not suffix:
                if scale:
                    scr1 = self._scale_score(  # type: ignore
                        score_key=key1, which="perturbed", inplace=False
                    )
                    scr2 = self._scale_score(  # type: ignore
                        score_key=key2, which="perturbed", inplace=False
                    )
                else:
                    scr1 = self.scores_rand[key1]
                    scr2 = self.scores_rand[key2]
            else:
                scr1 = self._get_perturbed_stats(score_key=key1, suffix=suffix)
                scr2 = self._get_perturbed_stats(score_key=key2, suffix=suffix)
                if scale:
                    scr1 = self._scale_score(  # type: ignore
                        score=scr1, which="perturbed", inplace=False
                    )
                    scr2 = self._scale_score(  # type: ignore
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
        groupby: str = "celltype",
        check: bool = True,
        topn: int = 500,
        simn: int = 1000,
        statistics: bool = True,
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
        rank_by: str = "up",
        **kwargs: Any,
    ) -> None:
        """
        Add scores to object based on marker genes for cell types from single cell data.
        Internally, scanpy.tl.rank_genes_groups is called on the linked AnnData object.
        Computed marker genes are used as scores and propagated.
        Optionally, statistics are calculated based on random permutations.

        :param groupby: column name in adata.obs with cell type labels
        :param check: whether to check the anndata object
        :param topn: use the topn marker genes per cell type (default: 500)
        :param simn: how many permutation to compute (default: 1000)
        :param statistics: whether to compute permutation results
        :param num_cores: number of cores to use
        :param log: logger
        :param rank_by: rank genes by upregulation ("up"), downregulation ("down") or absolute value ("abs")
        :param kwargs: arguments passed to `sc.tl.rank_genes_groups()`
        :return: None
        """
        self._check_init(check_adata=True)
        if check:
            if self.adata.X.max() > 50:  # type: ignore
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
            de_df = sc.get.rank_genes_groups_df(self.adata, group=None)  # type: ignore

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
        self,
        group_key: str = "celltype",
        score_key: str = "score",
        suffix: str = "",
        statistics: bool = True,
    ) -> None:
        """
        Combine differential expression based scores with other scores. E.g. combine them with GWAS based scores.

        :param group_key: column name of adata.obs with cell type labels that was used to derive marker genes
        :param score_key: score key to combine the DE scores with. E.g. may be a score key for a GWAS based score.
        :param suffix: suffix to use for combining scores. E.g. may be "__zscore" to combine zscores instead of directly combining the propagated scores.
        :param statistics: whether to calculate statistics based on random perturbation background.
        :return: None
        """
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

    def get_scores(
        self,
        which: str = "propagated",
        query: Optional[str] = None,
        sort_key: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Get scores.

        :param which: type ob scores to retrieve. Can be "original" (before propagation), "propagated" (after propagation) or "perturbed" (random permutations).
        :param query: pandas query string (`pd.DataFrame.query()`) that can be used to subset the returned data frame
        :param sort_key: column name that can be used to sort the returned data frame (in descending order)
        :param kwargs: options passed to `pd.filter(**kwargs)` for filtering columns of the returned data frame
        :return: a pandas data frame with selected scores
        """
        self._check_init()
        scores_out = {
            "propagated": self.scores_prop,
            "original": self.scores,
            "perturbed": self.scores_rand,
        }[which]

        if which == "perturbed":
            return copy.deepcopy(scores_out)  # type: ignore
        else:
            scores_out = scores_out.copy()  # type: ignore

        if query:
            scores_out = scores_out.query(query)  # type: ignore
        if sort_key:
            scores_out = scores_out.sort_values(sort_key, ascending=False)  # type: ignore
        if kwargs:
            scores_out = scores_out.filter(**kwargs)  # type: ignore

        return scores_out  # type: ignore

    def remove_scores(self, which: str = "propagated", **kwargs: Any) -> None:
        """
        Delete selected scores from the object.
        :param which: type ob scores to retrieve. Can be "original" (before propagation), "propagated" (after propagation) or "perturbed" (random permutations).
        :param kwargs: options passed to `pd.filter(**kwargs)` for selecting columns to DROP
        :return: None
        """
        self._check_init()
        if which == "perturbed":
            self.scores_rand = {}
        elif which == "all":
            self._init_scores()
        elif which == "propagated":
            if kwargs:
                cols = self.scores_prop.filter(**kwargs).columns  # type: ignore
                self.scores_prop = self.scores_prop.drop(columns=cols)  # type: ignore
            else:
                self.scores_prop = pd.DataFrame(index=list(self.grn.nodes))  # type: ignore
        elif which == "original":
            if kwargs:
                cols = self.scores.filter(**kwargs).columns  # type: ignore
                self.scores = self.scores.drop(columns=cols)  # type: ignore
            else:
                self.scores = pd.DataFrame(index=list(self.grn.nodes))  # type: ignore

    def get_components(self, sel_nodes: list[str]) -> tuple[nx.Graph, list[set]]:
        """
        Return connected components after subsetting the networkx graph.

        :param sel_nodes: nodes to select for subsetting
        :return: a tuple of the subsetted graph and a list of connected components (sets of nodes)
        """
        self._check_init()
        G_sub = self.grn.subgraph(sel_nodes)  # type: ignore
        cc = list(nx.connected_components(G_sub))
        print([len(x) for x in cc])
        return G_sub, cc

    def get_dfs(
        self, sel_nodes: Optional[list[str]] = None, **kwargs: Any
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return pandas data frames with nodes and edges.
        These can e.g. be used for import into Cytoscape or other software.

        :param sel_nodes: nodes for subsetting before creating data frames
        :param kwargs: options passed to `get_scores(**kwargs)`
        :return: a tuple of node table and edge table as pandas data frames
        """
        self._check_init()
        adj_df = pd.DataFrame(self.grn.edges, columns=["source", "target"])  # type: ignore
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
        self,
        std_cutoff: float = 1,
        row_pattern: str = ".*DE_(?P<rowname>.+?)__",
        **kwargs: Any,
    ) -> None:
        """
        Plot scores aggregated across all nodes.

        :param std_cutoff: cutoff on standard deviation for selecting nodes with larger variation across scores
        :param row_pattern: regex for extracting names for plotting from the score names (default: ".*DE_(?P<rowname>.+?)__")
        :param kwargs: options passed to `get_scores(**kwargs)`
        :return: None
        """
        """extract columns using like='' or regex='' and plot boxplot"""
        plt_df: pd.DataFrame
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff, :]
        # plt_df = plt_df.loc[~plt_df.apply(lambda x: all(x[0]==x), axis=1),:]
        plt_df = plt_df.apply(self._robust_z_score, axis=1)

        def rename_func(c):
            m = re.match(row_pattern, c)
            if not m:
                raise ValueError(f"could not match {c}")
            return m.group("rowname")

        plt_df = plt_df.rename(columns=rename_func)

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
        self,
        std_cutoff: float = 1,
        row_pattern: str = ".*DE_(?P<rowname>.+?)__",
        **kwargs: Any,
    ) -> None:
        """
        Plot scores.

        :param std_cutoff: cutoff on standard deviation for selecting nodes with larger variation across scores
        :param row_pattern: regex for extracting names for plotting from the score names (default: ".*DE_(?P<rowname>.+?)__")
        :param kwargs: options passed to `get_scores(**kwargs)`
        :return: None
        """
        plt_df: pd.DataFrame
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff, :]
        plt_df = plt_df.apply(self._std_scale, axis=1)

        def rename_func(c):
            m = re.match(row_pattern, c)
            if not m:
                raise ValueError(f"could not match {c}")
            return m.group("rowname")

        plt_df = plt_df.rename(columns=rename_func)

        plt_df = plt_df.loc[
            :, plt_df.median(axis=0).sort_values(ascending=False).index.tolist()
        ]
        rows = []
        for c in plt_df:
            rows.extend(plt_df[plt_df[c] == 1].index.tolist())
        plt_df = plt_df.loc[rows, :]

        sns.heatmap(plt_df, cmap="mako", yticklabels=False)
