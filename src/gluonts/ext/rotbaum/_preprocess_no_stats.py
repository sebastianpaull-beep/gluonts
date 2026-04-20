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

import logging
from itertools import chain
from typing import Dict, List

import numpy as np

from gluonts.ext.rotbaum._preprocess import PreprocessOnlyLagFeatures

logger = logging.getLogger(__name__)


class PreprocessOnlyLagFeaturesNoStats(PreprocessOnlyLagFeatures):
    """
    Like ``PreprocessOnlyLagFeatures``, but:

    - Target block: only the (optionally mean-centred) lag vector — no
      ``transform_dict`` stats (mean / std / n_lag_features / n_nans).
    - Dynamic real channels: only ``ent[0]`` per window — no ``ent[1]`` stats.

    ``dynamic_length`` is the context width only, matching the target lag block
    size expected by ``TreePredictor`` coordinate maps.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dynamic_length = self.context_window_size

    def make_features(self, time_series: Dict, starting_index: int) -> List:
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
        feat_dynamic_real = (
            list(
                chain(
                    *[
                        list(ent[0])
                        for ent in [
                            self._pre_transform(
                                ts[
                                    starting_index : end_index
                                    + self.forecast_horizon
                                ],
                                self.subtract_mean,
                                self.count_nans,
                            )
                            for ts in time_series["feat_dynamic_real"]
                        ]
                    ]
                )
            )
            if self.use_feat_dynamic_real
            else []
        )
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
        # Debugging aid: show exact block sizes when enabled.
        if logger.isEnabledFor(logging.INFO):
            try:
                logger.info(
                    "[RotbaumNoStats] feature_breakdown prefix=%s target_lags=%s "
                    "feat_static_real=%s feat_static_cat=%s "
                    "past_feat_dynamic_real=%s feat_dynamic_real=%s feat_dynamic_cat=%s "
                    "total=%s starting_index=%s",
                    len(prefix),
                    len(only_lag_features),
                    len(feat_static_real),
                    len(feat_static_cat),
                    len(past_feat_dynamic_real),
                    len(feat_dynamic_real),
                    len(feat_dynamic_cat),
                    len(feats),
                    starting_index,
                )
            except Exception:  # pragma: no cover
                pass
        return feats


def apply_rotbaum_no_stats_preprocess_patch() -> None:
    """Monkeypatch rotbaum so ``TreePredictor`` uses no-stats featurisation."""
    from gluonts.ext.rotbaum import _predictor as rot_predictor
    from gluonts.ext.rotbaum import _preprocess as rot_preprocess

    rot_predictor.PreprocessOnlyLagFeatures = PreprocessOnlyLagFeaturesNoStats
    rot_preprocess.PreprocessOnlyLagFeatures = PreprocessOnlyLagFeaturesNoStats
