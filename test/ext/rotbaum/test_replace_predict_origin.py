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

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.ext.rotbaum import (
    TreeEstimator,
    apply_rotbaum_no_stats_preprocess_patch,
)
from gluonts.ext.rotbaum import _predictor as rot_predictor
from gluonts.ext.rotbaum import _preprocess as rot_preprocess


@pytest.fixture
def restore_preprocess_only_lag_features():
    """Undo global monkeypatch after each test."""
    orig_pred = rot_predictor.PreprocessOnlyLagFeatures
    orig_pre = rot_preprocess.PreprocessOnlyLagFeatures
    yield
    rot_predictor.PreprocessOnlyLagFeatures = orig_pred
    rot_preprocess.PreprocessOnlyLagFeatures = orig_pre


def _oracle_train_test_pair(
    horizon: int, context: int, rng: np.random.Generator
) -> tuple[dict, dict]:
    """Train/test entries mirroring GluonTS + Doctorate oracle split."""
    t_len = 80
    y = rng.normal(size=t_len).astype(np.float32)
    train_target = y[:-horizon]
    train_fdr = np.concatenate([train_target, y[-horizon:]]).astype(np.float32)
    test_fdr = np.concatenate([y, y[-horizon:]]).astype(np.float32)
    past_row = y[:-horizon]
    train = {
        FieldName.TARGET: train_target.tolist(),
        FieldName.START: pd.Period("2020-01-06", freq="W-MON"),
        FieldName.FEAT_DYNAMIC_REAL: [train_fdr.tolist()],
        FieldName.PAST_FEAT_DYNAMIC_REAL: [past_row.tolist()],
    }
    test = {
        FieldName.TARGET: y.tolist(),
        FieldName.START: pd.Period("2020-01-06", freq="W-MON"),
        FieldName.FEAT_DYNAMIC_REAL: [test_fdr.tolist()],
        FieldName.PAST_FEAT_DYNAMIC_REAL: [past_row.tolist()],
    }
    return train, test


def _rows_match(predict_row, row) -> bool:
    if len(predict_row) != len(row):
        return False
    return bool(np.allclose(predict_row, row, atol=1e-9, rtol=0))


def _predict_row_in_train_matrix(predictor, test_entry, horizon: int) -> bool:
    preprocess = predictor.preprocess_object
    context = preprocess.context_window_size
    target = np.asarray(test_entry["target"], dtype=float)
    predict_start = len(target) - context
    per_h = preprocess.feature_data_per_horizon
    for h in range(horizon):
        predict_row = preprocess.make_features(test_entry, predict_start, h)
        if not any(_rows_match(predict_row, row) for row in per_h[h]):
            return False
    return True


def _last_train_row_is_predict_row(predictor, test_entry, horizon: int) -> bool:
    preprocess = predictor.preprocess_object
    context = preprocess.context_window_size
    target = np.asarray(test_entry["target"], dtype=float)
    predict_start = len(target) - context
    per_h = preprocess.feature_data_per_horizon
    last_idx = -1
    for h in range(horizon):
        predict_row = preprocess.make_features(test_entry, predict_start, h)
        if not _rows_match(predict_row, per_h[h][last_idx]):
            return False
    return True


def _base_estimator_kwargs(horizon: int, context: int) -> dict:
    return dict(
        freq="W-MON",
        prediction_length=horizon,
        context_length=context,
        method="QuantileRegression",
        quantiles=[0.5],
        model_params={
            "n_estimators": 5,
            "learning_rate": 0.1,
            "num_leaves": 8,
            "n_jobs": 1,
            "random_state": 0,
        },
        use_feat_static_real=False,
        use_past_feat_dynamic_real=True,
        use_feat_dynamic_real=True,
        subtract_mean=False,
        count_nans=False,
    )


def test_replace_predict_origin_last_row_is_predict_row(
    restore_preprocess_only_lag_features,
):
    """Without replace: predict row absent; with replace: last row matches."""
    apply_rotbaum_no_stats_preprocess_patch()
    horizon = 4
    context = 1
    rng = np.random.default_rng(42)
    train, test = _oracle_train_test_pair(horizon, context, rng)
    train_ds = ListDataset([train], freq="W-MON")
    test_ds = ListDataset([test], freq="W-MON")

    predictor_baseline = TreeEstimator(**_base_estimator_kwargs(horizon, context)).train(
        training_data=train_ds
    )
    assert not _predict_row_in_train_matrix(predictor_baseline, test, horizon)

    predictor_fixed = TreeEstimator(**_base_estimator_kwargs(horizon, context)).train(
        training_data=train_ds,
        validation_dataset=test_ds,
        replace_predict_origin=True,
    )
    assert _last_train_row_is_predict_row(predictor_fixed, test, horizon)
    assert _predict_row_in_train_matrix(predictor_fixed, test, horizon)


def test_replace_predict_origin_no_extra_rows(
    restore_preprocess_only_lag_features,
):
    """replace_predict_origin overwrites last row; does not append extra rows."""
    apply_rotbaum_no_stats_preprocess_patch()
    horizon = 4
    context = 1
    rng = np.random.default_rng(42)
    train, test = _oracle_train_test_pair(horizon, context, rng)
    train_ds = ListDataset([train], freq="W-MON")
    test_ds = ListDataset([test], freq="W-MON")
    base_kwargs = _base_estimator_kwargs(horizon, context)

    predictor_fixed = TreeEstimator(**base_kwargs).train(
        training_data=train_ds,
        validation_dataset=test_ds,
        replace_predict_origin=True,
    )
    expected_windows = len(train["target"]) - context - horizon + 1
    assert len(predictor_fixed.preprocess_object.target_data) == expected_windows


def test_replace_predict_origin_flag_rename(
    restore_preprocess_only_lag_features,
):
    """TreeEstimator accepts replace_predict_origin; append_predict_origin rejected."""
    apply_rotbaum_no_stats_preprocess_patch()
    horizon = 4
    context = 1
    rng = np.random.default_rng(42)
    train, test = _oracle_train_test_pair(horizon, context, rng)
    train_ds = ListDataset([train], freq="W-MON")
    test_ds = ListDataset([test], freq="W-MON")
    estimator = TreeEstimator(**_base_estimator_kwargs(horizon, context))

    predictor = estimator.train(
        training_data=train_ds,
        validation_dataset=test_ds,
        replace_predict_origin=True,
    )
    assert predictor is not None

    with pytest.raises(TypeError):
        estimator.train(
            training_data=train_ds,
            validation_dataset=test_ds,
            append_predict_origin=True,
        )
