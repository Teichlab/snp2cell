import copy
import gc
import logging
import multiprocessing
import os
import pickle
import random
import re
from pathlib import Path
from enum import Enum
from typing import Union, Optional, Any, Iterable, Dict, List, Tuple

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

RAW_COUNT_THR = 50
MAD_SCALE = 0.6745
NCPU = multiprocessing.cpu_count()


class SUFFIX(Enum):
    ZSCORE = "__zscore"
    ZSCORE_MAD = "__zscore_mad"


class SNP2CELL:
    def __init__(self, path: Optional[Union[str, os.PathLike]] = None) -> None:
        """
        Initialize the SNP2CELL object.

        Parameters:
        path (Optional[Union[str, os.PathLike]]): Path to load data from.
        """
        self.grn: Optional[nx.Graph] = None
        self.adata: Optional[sc.AnnData] = None
        self.scores: Optional[pd.DataFrame] = None
        self.scores_prop: Optional[pd.DataFrame] = None
        self.scores_rand: Dict[str, pd.DataFrame] = {}
        self.de_groups: Dict[str, List[str]] = {}

        if path:
            self.load_data(path)

    def __repr__(self) -> str:
        """
        Return a string representation of the SNP2CELL object.

        Returns
        -------
        str
            String representation of the object.
        """
        return (
            f"GRN: {self.grn}\n"
            f"original scores: {self.scores.shape if self.scores is not None else 'None'}\n"
            f"propagated scores: {self.scores_prop.shape if self.scores_prop is not None else 'None'}\n"
            f"score perturbations for: {list(self.scores_rand) if self.scores_rand else 'None'}\n"
            f"anndata: {self.adata.shape if self.adata is not None else 'None'}"
        )

    ###--------------------------------------------------- private

    def _init_scores(self) -> None:
        """
        Initialize scores DataFrames and related attributes.

        Raises
        ------
        ValueError
            If no GRN is set.
        """
        if self.grn:
            self.scores = pd.DataFrame(index=list(self.grn.nodes))
            self.scores_prop = pd.DataFrame(index=list(self.grn.nodes))
            self.scores_rand = {}
            self.de_groups = {}
        else:
            raise ValueError("No GRN set, add GRN first.")

    def _set_grn(self, nx_grn: nx.Graph) -> None:
        """
        Set the GRN (Gene Regulatory Network).

        Parameters
        ----------
        nx_grn : nx.Graph
            The networkx graph representing the GRN.
        """
        self.grn = nx_grn.to_undirected()

    def _add_de_groups(self, groupby: str, groups: List[str]) -> None:
        """
        Add differential expression groups.

        Parameters
        ----------
        groupby : str
            The key to group by.
        groups : List[str]
            The list of groups.

        Raises
        ------
        ValueError
            If the group key already exists or if groups already exist in another key.
        """
        if groupby in self.de_groups:
            raise ValueError(f"group key {groupby} already exists")

        for k, v in self.de_groups.items():
            shared_keys = set(groups) & set(v)
            if shared_keys:
                raise ValueError(f"Groups {shared_keys} already exist in {k}")
        
        self.de_groups[groupby] = groups

    def _scale_score(
        self,
        score_key: Optional[str] = None,
        score: Optional[Union[pd.Series, pd.DataFrame]] = None,
        which: str = "original",
        inplace: bool = True,
    ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        """
        Scale scores.

        Parameters
        ----------
        score_key : Optional[str], optional
            Key of the score to scale, by default None.
        score : Optional[Union[pd.Series, pd.DataFrame]], optional
            Score to scale, by default None.
        which : str, optional
            Type of score to scale ("original", "propagated", "perturbed"), by default "original".
        inplace : bool, optional
            Whether to modify the scores in place, by default True.

        Returns
        -------
        Optional[Union[pd.Series, pd.DataFrame]]
            Scaled scores if inplace is False, otherwise None.

        Raises
        ------
        ValueError
            If neither score_key nor score is provided or if scores are not initialized.
        """
        if score_key is None and score is None:
            raise ValueError("Either `score_key` or `score` must be provided.")
        scores_out: Optional[Union[pd.DataFrame, Dict[str, pd.DataFrame]]]
        scores_out = {
            "propagated": self.scores_prop,
            "original": self.scores,
            "perturbed": self.scores_rand,
        }[which]
        if scores_out is None:
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

    def _get_perturbed_stats(self, score_key: str, suffix: SUFFIX) -> pd.DataFrame:
        """
        Get perturbed statistics.

        Parameters
        ----------
        score_key : str
            Key of the score.
        suffix : SUFFIX
            Suffix indicating the type of statistics.

        Returns
        -------
        pd.DataFrame
            DataFrame with perturbed statistics.

        Raises
        ------
        ValueError
            If the suffix is invalid.
        """
        if suffix not in [s.value for s in SUFFIX]:
            raise ValueError(
                f"Invalid suffix. Must be one of {[s.value for s in SUFFIX]}."
            )
        score = self.scores_rand[score_key]
        if suffix == SUFFIX.ZSCORE:
            score = score.apply(self._z_score, axis=0)
        if suffix == SUFFIX.ZSCORE_MAD:
            score = score.apply(self._robust_z_score, axis=0)
        return score

    @staticmethod
    def _robust_z_score(series: pd.Series) -> pd.Series:
        """
        Calculate robust z-score.

        Parameters
        ----------
        series : pd.Series
            Series to calculate z-score for.

        Returns
        -------
        pd.Series
            Series with robust z-scores.
        """
        mad = sp.stats.median_abs_deviation(series, scale=1.0)
        zscore = MAD_SCALE * (series - series.median()) / mad
        return zscore

    @staticmethod
    def _z_score(series: pd.Series) -> pd.Series:
        """
        Calculate z-score.

        Parameters
        ----------
        series : pd.Series
            Series to calculate z-score for.

        Returns
        -------
        pd.Series
            Series with z-scores.
        """
        return (series - series.mean()) / series.std()

    @staticmethod
    def _std_scale(series: pd.Series) -> pd.Series:
        """
        Standard scale a series.

        Parameters
        ----------
        series : pd.Series
            Series to scale.

        Returns
        -------
        pd.Series
            Scaled series.
        """
        return (series - series.min()) / (series.max() - series.min())

    def _prop_scr(self, scr_dct: Dict[str, float]) -> Dict[str, float]:
        """
        Propagate scores using PageRank.

        Parameters
        ----------
        scr_dct : Dict[str, float]
            Dictionary of scores.

        Returns
        -------
        Dict[str, float]
            Dictionary of propagated scores.
        """
        return nx.pagerank(
            self.grn,
            personalization=scr_dct,
        )

    def _defrag_pandas(self) -> None:
        """
        Defragment pandas DataFrames to optimize memory usage.

        Raises
        ------
        ValueError
            If scores, propagated scores, or random scores are not initialized.
        """
        if self.scores is None or self.scores_prop is None or not self.scores_rand:
            raise ValueError(
                "Scores, propagated scores, and random scores must be initialized."
            )
        self.scores = self.scores.copy()
        self.scores_prop = self.scores_prop.copy()
        for k in self.scores_rand.keys():
            self.scores_rand[k] = self.scores_rand[k].copy()
        gc.collect()

    def _check_init(self, check_adata: bool = False) -> None:
        """
        Check if the object is initialized.

        Parameters
        ----------
        check_adata : bool, optional
            Whether to check if AnnData is initialized, by default False.

        Raises
        ------
        ValueError
            If scores, propagated scores, random scores, or GRN are not initialized.
        """
        if self.scores is None or self.scores_prop is None or self.scores_rand is None:
            raise ValueError("Need to init scores first.")
        if self.grn is None:
            raise ValueError("Need to set GRN first.")
        if check_adata and self.adata is None:
            raise ValueError("Need to add AnnData first")

    ###--------------------------------------------------- input/output

    def save_obj(self, path: Union[str, os.PathLike]) -> None:
        """
        Deprecated, use `SNP2CELL.save_data()` instead.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            Path to save the object.
        """
        with open(path, "wb") as f:
            dill.dump(self, f)

    def save_data(self, path: Union[str, os.PathLike]) -> None:
        """
        Save object data. Reload using `s2c = SNP2CELL(<path>)`.

        Parameters
        ----------
        path : Union[str, os.PathLike]
            Output path for pickle file.
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

        Parameters
        ----------
        path : Union[str, os.PathLike]
            Input path for pickle file with data.
        overwrite : bool, optional
            Whether to overwrite existing data, by default False.

        Raises
        ------
        IndexError
            If existing data is found and overwrite is False.
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

    def add_grn_from_pandas(self, adjacency_df: pd.DataFrame) -> None:
        """
        Add GRN from pandas DataFrame.

        Parameters
        ----------
        adjacency_df : pd.DataFrame
            DataFrame with adjacency information.
        """
        raise NotImplementedError("This method is not yet implemented.")

    def add_grn_from_networkx(self, nx_grn: nx.Graph, overwrite: bool = False) -> None:
        """
        Add GRN from networkx object to snp2cell object.

        Parameters
        ----------
        nx_grn : nx.Graph
            Networkx object.
        overwrite : bool, optional
            Whether to overwrite existing networkx object, by default False.

        Raises
        ------
        IndexError
            If existing scores are found and overwrite is False.
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

        Parameters
        ----------
        adata : sc.AnnData
            Scanpy AnnData object.
        overwrite : bool, optional
            Whether to overwrite existing AnnData object, by default False.

        Raises
        ------
        IndexError
            If linked AnnData is found and overwrite is False.
        """
        if self.adata is not None and not overwrite:
            raise IndexError("linked adata found, set overwrite=True to replace.")
        self.adata = adata

    ###--------------------------------------------------- scores

    @add_logger(show_start_end=False)
    def add_score(
        self,
        score_dct: Dict[str, float],
        score_key: str = "score",
        propagate: bool = True,
        statistics: bool = True,
        num_rand: int = 1000,
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Add a score to the object. Optionally propagate the score and calculate permutation statistics.

        Parameters
        ----------
        score_dct : Dict[str, float]
            Dictionary with score values.
        score_key : str, optional
            Key under which to store the scores, by default "score".
        propagate : bool, optional
            Whether to propagate the scores, by default True.
        statistics : bool, optional
            Whether to calculate statistics (only if propagate is True), by default True.
        num_rand : int, optional
            Number of permutations to compute if statistics is True, by default 1000.
        num_cores : Optional[int], optional
            Number of cores to use, by default None.
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
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
                self.rand_sim(score_key=score_key, num_cores=num_cores, n=num_rand)
                self.add_score_statistics(score_keys=score_key)
        self._defrag_pandas()

    def propagate_score(self, score_key: str = "score") -> Tuple[str, Dict[str, float]]:
        """
        Propagate score.

        Parameters
        ----------
        score_key : str, optional
            Key under which the score is stored, by default "score".

        Returns
        -------
        Tuple[str, Dict[str, float]]
            Tuple of the score key and dict with score values after propagation.
        """
        self._check_init()
        scr_dct = self.scores[score_key].to_dict()  # type: ignore
        p_scr_dct = self._prop_scr(scr_dct)
        return score_key, p_scr_dct

    @add_logger()
    def propagate_scores(
        self,
        score_keys: List[str],
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Propagate multiple scores.

        Parameters
        ----------
        score_keys : List[str]
            Keys under which the scores are stored.
        num_cores : Optional[int], optional
            Number of cores to use, by default None.
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
        """
        self._check_init()
        use_scores = set(score_keys)
        for k in score_keys:
            if self.scores[k].sum() == 0:
                use_scores -= {k}
                log.warning(f"score {k} is all zero, skipping")
        score_keys = list(use_scores)

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
        score_key: Union[str, List[str]] = "score",
        perturb_key: Optional[str] = None,
        n: int = 1000,
        num_cores: Optional[int] = None,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Create random permutations and propagate them.

        Parameters
        ----------
        score_key : Union[str, List[str]], optional
            Key of score for which to compute statistics, by default "score".
        perturb_key : Optional[str], optional
            Key under which to store permutation results, by default None.
        n : int, optional
            Number of permutation to create, by default 1000.
        num_cores : Optional[int], optional
            Number of cores to use, by default None.
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
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
        score_keys: Union[str, List[str], Dict[str, str]] = "score",
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Calculate statistics from random permutations for scores.

        Parameters
        ----------
        score_keys : Union[str, List[str], Dict[str, str]], optional
            Scores for which to calculate statistics. May be a single score, a list of scores or a dictionary ({<score_key>: <perturb_key>}). If not a dictionary assuming `score_key==perturb_key`, by default "score".
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
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

        concat_dfs: List[pd.DataFrame] = [self.scores_prop] + dfs  # type: ignore
        self.scores_prop = pd.concat(concat_dfs, axis=1)
        self._defrag_pandas()

    @add_logger()
    def combine_scores(
        self,
        combine: Iterable[Tuple[str, str]],
        scale: bool = False,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Combine pairs of scores. (E.g. for combining scores from GWAS data and differential expression).
        Currently, the only implemented option for combining the scores is min(score1, score2).
        New scores will be added, called `min(<key1>,<key2>)` for each score pair.

        Parameters
        ----------
        combine : Iterable[Tuple[str, str]]
            A list of pairs of scores to combine.
        scale : bool, optional
            Whether to scale scores between 0 and 1 before combining, by default False.
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
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
        combine: Iterable[Tuple[str, str, str]],
        scale: bool = False,
        log: logging.Logger = logging.getLogger(),
    ) -> None:
        """
        Create random permutation background for combined scores.
        Currently, the only implemented option for combining the scores is min(score1, score2).

        Parameters
        ----------
        combine : Iterable[Tuple[str, str, str]]
            List of score pairs for which to create random permutation background. This is a list of triples, also containing an optional suffix for each pair. Currently, the only allowed suffix is "__zscore" (which will combine zscores instead of using permuted scores directly).
        scale : bool, optional
            Whether to scale scores between 0 and 1 before combining, by default False.
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
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
        pval: float = 0.05,
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

        Parameters
        ----------
        groupby : str, optional
            Column name in adata.obs with cell type labels, by default "celltype".
        check : bool, optional
            Whether to check the anndata object, by default True.
        topn : int, optional
            Use the topn marker genes per cell type, by default 500.
        pval : float, optional
            P-value threshold for marker genes, by default 0.05.
        simn : int, optional
            How many permutation to compute, by default 1000.
        statistics : bool, optional
            Whether to compute permutation results, by default True.
        num_cores : Optional[int], optional
            Number of cores to use, by default None.
        log : logging.Logger, optional
            Logger, by default logging.getLogger().
        rank_by : str, optional
            Rank genes by upregulation ("up"), downregulation ("down") or absolute value ("abs"), by default "up".
        kwargs : Any
            Arguments passed to `sc.tl.rank_genes_groups()`.
        """
        self._check_init(check_adata=True)
        if check:
            if self.adata.X.max() > RAW_COUNT_THR:  # type: ignore
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
                query_str = f"group == '{grp}' and pvals_adj < {pval} and scores > 0"
            scr = de_df.query(query_str).set_index("names")["scores"][:topn].to_dict()

            scr_key = f"DE_{grp}__score"

            self.add_score(scr, score_key=scr_key, propagate=False, statistics=False)

            if self.scores[scr_key].sum() == 0:
                log.warning(f"score for '{grp}' ('{groupby}') is all zero, ignoring")
                self.de_groups[groupby].remove(grp)
            else:
                score_keys.append(scr_key)

        self.propagate_scores(score_keys, num_cores=num_cores)

        if statistics:
            self.rand_sim(
                score_key=score_keys,
                perturb_key=f"DE_{groupby}__score",
                n=simn,
                num_cores=num_cores,
            )

            self.add_score_statistics(
                score_keys={k: f"DE_{groupby}__score" for k in score_keys}
            )
        self._defrag_pandas()

    def adata_combine_de_scores(
        self,
        group_key: str = "celltype",
        score_key: str = "score",
        suffix: str = "",
        scale: bool = False,
        statistics: bool = True,
    ) -> None:
        """
        Combine differential expression based scores with other scores. E.g. combine them with GWAS based scores.

        Parameters
        ----------
        group_key : str, optional
            Column name of adata.obs with cell type labels that was used to derive marker genes, by default "celltype".
        score_key : str, optional
            Score key to combine the DE scores with. E.g. may be a score key for a GWAS based score, by default "score".
        suffix : str, optional
            Suffix to use for combining scores. E.g. may be "__zscore" to combine zscores instead of directly combining the propagated scores, by default "".
        scale : bool, optional
            Whether to scale scores between 0 and 1 before combining, by default False.
        statistics : bool, optional
            Whether to calculate statistics based on random perturbation background, by default True.
        """
        groups = self.de_groups[group_key]

        combine = [
            (f"DE_{grp}__score{suffix}", f"{score_key}{suffix}") for grp in groups
        ]
        self.combine_scores(combine, scale=scale)

        if statistics:
            self.combine_scores_rand(
                [(f"DE_{group_key}__score", score_key, suffix)], scale=scale
            )

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

        Parameters
        ----------
        which : str, optional
            Type of scores to retrieve. Can be "original" (before propagation), "propagated" (after propagation) or "perturbed" (random permutations), by default "propagated".
        query : Optional[str], optional
            Pandas query string (`pd.DataFrame.query()`) that can be used to subset the returned data frame, by default None.
        sort_key : Optional[str], optional
            Column name that can be used to sort the returned data frame (in descending order), by default None.
        kwargs : Any
            Options passed to `pd.filter(**kwargs)` for filtering columns of the returned data frame.

        Returns
        -------
        pd.DataFrame
            A pandas data frame with selected scores.
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

        Parameters
        ----------
        which : str, optional
            Type of scores to retrieve. Can be "original" (before propagation), "propagated" (after propagation) or "perturbed" (random permutations), by default "propagated".
        kwargs : Any
            Options passed to `pd.filter(**kwargs)` for selecting columns to DROP.
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
                for k in self.de_groups:
                    self.de_groups[k] = [i for i in self.de_groups[k] if i not in cols]
            else:
                self.scores = pd.DataFrame(index=list(self.grn.nodes))  # type: ignore
                self.de_groups = {}

    def get_components(self, sel_nodes: List[str]) -> Tuple[nx.Graph, List[set]]:
        """
        Return connected components after subsetting the networkx graph.

        Parameters
        ----------
        sel_nodes : List[str]
            Nodes to select for subsetting.

        Returns
        -------
        Tuple[nx.Graph, List[set]]
            A tuple of the subsetted graph and a list of connected components (sets of nodes).
        """
        self._check_init()
        G_sub = self.grn.subgraph(sel_nodes)  # type: ignore
        cc = list(nx.connected_components(G_sub))
        print([len(x) for x in cc])
        return G_sub, cc

    def get_dfs(
        self, sel_nodes: Optional[List[str]] = None, **kwargs: Any
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Return pandas data frames with nodes and edges.
        These can e.g. be used for import into Cytoscape or other software.

        Parameters
        ----------
        sel_nodes : Optional[List[str]]
            Nodes for subsetting before creating data frames.
        kwargs : Any
            Options passed to `get_scores(**kwargs)`.

        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple of node table and edge table as pandas data frames.
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

    @staticmethod
    def rename_column(c: str, row_pattern: str = ".*DE_(?P<rowname>.+?)__") -> str:
        """
        Rename column based on a pattern.

        Parameters
        ----------
        c : str
            Column name.
        row_pattern : str, optional
            Regex pattern for extracting names, by default ".*DE_(?P<rowname>.+?)__".

        Returns
        -------
        str
            Renamed column.

        Raises
        ------
        ValueError
            If the pattern does not match.
        """
        m = re.match(row_pattern, c)
        if not m:
            raise ValueError(f"could not match {c}")
        return m.group("rowname")

    def plot_group_summary(
        self,
        std_cutoff: float = 1,
        row_pattern: str = ".*DE_(?P<rowname>.+?)__",
        **kwargs: Any,
    ) -> None:
        """
        Plot scores aggregated across all nodes.

        Parameters
        ----------
        std_cutoff : float, optional
            Cutoff on standard deviation for selecting nodes with larger variation across scores, by default 1.
        row_pattern : str, optional
            Regex for extracting names for plotting from the score names, by default ".*DE_(?P<rowname>.+?)__".
        kwargs : Any
            Options passed to `get_scores(**kwargs)`.

        Returns
        -------
        None
        """
        plt_df: pd.DataFrame
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff, :]
        # plt_df = plt_df.loc[~plt_df.apply(lambda x: all(x[0]==x), axis=1),:]
        plt_df = plt_df.apply(self._robust_z_score, axis=1)

        plt_df = plt_df.rename(columns=self.rename_column)

        sns.set_theme(style="ticks")
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

        Parameters
        ----------
        std_cutoff : float, optional
            Cutoff on standard deviation for selecting nodes with larger variation across scores, by default 1.
        row_pattern : str, optional
            Regex for extracting names for plotting from the score names, by default ".*DE_(?P<rowname>.+?)__".
        kwargs : Any
            Options passed to `get_scores(**kwargs)`.

        Returns
        -------
        None
        """
        plt_df: pd.DataFrame
        plt_df = self.get_scores(**kwargs)
        plt_df = plt_df.loc[plt_df.std(axis=1) > std_cutoff, :]
        plt_df = plt_df.apply(self._std_scale, axis=1)

        plt_df = plt_df.rename(columns=self.rename_column)

        plt_df = plt_df.loc[
            :, plt_df.median(axis=0).sort_values(ascending=False).index.tolist()
        ]
        rows = []
        for c in plt_df:
            rows.extend(plt_df[plt_df[c] == 1].index.tolist())
        plt_df = plt_df.loc[rows, :]

        sns.heatmap(plt_df, cmap="mako", yticklabels=False)
