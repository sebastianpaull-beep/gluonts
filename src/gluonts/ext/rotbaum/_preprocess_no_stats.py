# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""Rotbaum preprocessor variant: raw lag windows only, no summary-stat columns."""

import concurrent.futures
import logging
from itertools import chain
from typing import Dict, List, Tuple

import numpy as np

from gluonts.ext.rotbaum._preprocess import PreprocessOnlyLagFeatures

logger = logging.getLogger(__name__)


def _preprocess_no_stats_ts_worker(
    args: Tuple[PreprocessOnlyLagFeatures, Dict],
) -> Tuple[List[List], List]:
    """Picklable worker for ``ProcessPoolExecutor`` (single-arg tuple)."""
    preprocess, ts = args
    return preprocess._preprocess_single_ts_all_horizons(ts)


class PreprocessOnlyLagFeaturesNoStats(PreprocessOnlyLagFeatures):
    """
    Like ``PreprocessOnlyLagFeatures``, but:

    - Target block: only the (optionally mean-centred) lag vector — no
      ``transform_dict`` stats (mean / std / n_lag_features / n_nans).
    - Dynamic real channels: only ``ent[0]`` per window — no ``ent[1]`` stats.
    - ``feat_dynamic_real``: one value per channel aligned with
      ``horizon_index`` (see ``make_features``).

    ``dynamic_length`` is the context width only, matching the target lag block
    size expected by ``TreePredictor`` coordinate maps.

    ``feat_dynamic_real`` is sliced per forecast step: callers must pass
    ``horizon_index`` to ``make_features``. Training collates
    ``feature_data_per_horizon[h]`` for each step model.
    """

    #: ``TreePredictor`` / ``explain`` branch on this instead of ``isinstance``.
    slices_feat_dynamic_real_per_horizon = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_length = self.context_window_size

    def make_features(
        self, time_series: Dict, starting_index: int, horizon_index: int
    ) -> List:
        end_index = starting_index + self.context_window_size
        if starting_index < 0:
            prefix = [None] * abs(starting_index)
        else:
            prefix = []
        time_series_window = time_series["target"][starting_index:end_index]
        only_lag_features, _transform_dict = self._pre_transform(
            time_series_window, self.subtract_mean, self.count_nans
        )

        feat_static_real = (
            list(time_series["feat_static_real"])
            if self.use_feat_static_real
            else []
        )
        if self.cardinality:
            feat_static_cat = (
                self.encode_one_hot_all(time_series["feat_static_cat"])
                if self.one_hot_encode
                else list(time_series["feat_static_cat"])
            )
        else:
            feat_static_cat = []

        past_feat_dynamic_real = (
            list(
                chain(
                    *[
                        list(ent[0])
                        for ent in [
                            self._pre_transform(
                                ts[starting_index:end_index],
                                self.subtract_mean,
                                self.count_nans,
                            )
                            for ts in time_series["past_feat_dynamic_real"]
                        ]
                    ]
                )
            )
            if self.use_past_feat_dynamic_real
            else []
        )
        feat_dynamic_real = []
        if self.use_feat_dynamic_real:
            idx = self.context_window_size + horizon_index
            for ts in time_series["feat_dynamic_real"]:
                ent = self._pre_transform(
                    ts[starting_index : end_index + self.forecast_horizon],
                    self.subtract_mean,
                    self.count_nans,
                )
                lag_vec = np.asarray(ent[0], dtype=float)
                if idx < 0 or idx >= lag_vec.shape[0]:
                    raise IndexError(
                        "feat_dynamic_real horizon slice out of bounds: "
                        f"idx={idx}, len={lag_vec.shape[0]}, "
                        f"context_window_size={self.context_window_size}, "
                        f"horizon_index={horizon_index}"
                    )
                feat_dynamic_real.append(float(lag_vec[idx]))
        feat_dynamic_cat = (
            [
                elem
                for ent in time_series["feat_dynamic_cat"]
                for elem in ent[starting_index:end_index]
            ]
            if self.use_feat_dynamic_cat
            else []
        )

        np_feat_static_cat = np.array(feat_static_cat)
        assert (not feat_static_cat) or all(
            np.floor(np_feat_static_cat) == np_feat_static_cat
        )

        np_feat_dynamic_cat = np.array(feat_dynamic_cat)
        assert (not feat_dynamic_cat) or all(
            np.floor(np_feat_dynamic_cat) == np_feat_dynamic_cat
        )

        feat_dynamics = (
            past_feat_dynamic_real + feat_dynamic_real + feat_dynamic_cat
        )
        feat_statics = feat_static_real + feat_static_cat
        only_lag_features = list(only_lag_features)
        feats = prefix + only_lag_features + feat_statics + feat_dynamics
        if logger.isEnabledFor(logging.INFO):
            try:
                logger.info(
                    "[RotbaumNoStats] feature_breakdown prefix=%s target_lags=%s "
                    "feat_static_real=%s feat_static_cat=%s "
                    "past_feat_dynamic_real=%s feat_dynamic_real=%s feat_dynamic_cat=%s "
                    "total=%s starting_index=%s horizon_index=%s",
                    len(prefix),
                    len(only_lag_features),
                    len(feat_static_real),
                    len(feat_static_cat),
                    len(past_feat_dynamic_real),
                    len(feat_dynamic_real),
                    len(feat_dynamic_cat),
                    len(feats),
                    starting_index,
                    horizon_index,
                )
            except Exception:  # pragma: no cover
                pass
        return feats

    def _preprocess_single_ts_all_horizons(
        self, time_series: Dict
    ) -> Tuple[List[List], List]:
        """
        For each context window, build one feature row per horizon index and
        one target vector over the forecast horizon.
        """
        if self.stratify_targets:
            raise ValueError(
                "PreprocessOnlyLagFeaturesNoStats does not support "
                "stratify_targets=True"
            )
        altered_time_series = time_series.copy()
        if self.n_ignore_last > 0:
            altered_time_series["target"] = altered_time_series["target"][
                : -self.n_ignore_last
            ]
        rows_per_h = [[] for _ in range(self.forecast_horizon)]
        target_data = []
        max_num_context_windows = (
            len(altered_time_series["target"])
            - self.context_window_size
            - self.forecast_horizon
            + 1
        )
        if max_num_context_windows < 1:
            return [[] for _ in range(self.forecast_horizon)], [[]]

        if self.num_samples > 0:
            locations = [
                np.random.randint(max_num_context_windows)
                for _ in range(self.num_samples)
            ]
        else:
            locations = range(max_num_context_windows)
        for starting_index in locations:
            for h in range(self.forecast_horizon):
                rows_per_h[h].append(
                    self.make_features(altered_time_series, starting_index, h)
                )
            target_data.append(
                time_series["target"][
                    starting_index
                    + self.context_window_size : starting_index
                    + self.context_window_size
                    + self.forecast_horizon
                ]
            )
        return rows_per_h, target_data

    def preprocess_from_list(
        self, ts_list, change_internal_variables: bool = True
    ) -> Tuple:
        feature_data_per_h = [[] for _ in range(self.forecast_horizon)]
        target_data: List = []
        self.num_samples = self.get_num_samples(ts_list)

        if isinstance(self.cardinality, str):
            self.cardinality = (
                self.infer_cardinalities(ts_list)
                if self.cardinality == "auto"
                else []
            )

        self.infer_feature_characteristics(ts_list[0])

        with concurrent.futures.ProcessPoolExecutor() as executor:
            logger.info("using concurrent data preprocessing (no-stats per-h)")
            for result in executor.map(
                _preprocess_no_stats_ts_worker,
                [(self, ts) for ts in ts_list],
            ):
                ts_rows_per_h, ts_target_data = result
                if len(ts_target_data) > 0 and len(ts_target_data[0]) > 0:
                    for h in range(self.forecast_horizon):
                        feature_data_per_h[h].extend(ts_rows_per_h[h])
                    target_data.extend(ts_target_data)

        logging.info(
            "Done preprocessing. Resulting number of datapoints is: {}".format(
                len(target_data)
            )
        )
        if change_internal_variables:
            self.feature_data_per_horizon = feature_data_per_h
            self.feature_data = None
            self.target_data = target_data
        else:
            return feature_data_per_h, target_data


def apply_rotbaum_no_stats_preprocess_patch() -> None:
    """Monkeypatch rotbaum so ``TreePredictor`` uses no-stats featurisation."""
    from gluonts.ext.rotbaum import _predictor as rot_predictor
    from gluonts.ext.rotbaum import _preprocess as rot_preprocess

    rot_predictor.PreprocessOnlyLagFeatures = PreprocessOnlyLagFeaturesNoStats
    rot_preprocess.PreprocessOnlyLagFeatures = PreprocessOnlyLagFeaturesNoStats
