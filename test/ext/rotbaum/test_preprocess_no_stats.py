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
from gluonts.ext.rotbaum._preprocess import PreprocessOnlyLagFeatures
from gluonts.ext.rotbaum._preprocess_no_stats import (
    PreprocessOnlyLagFeaturesNoStats,
)


@pytest.fixture
def restore_preprocess_only_lag_features():
    """Undo global monkeypatch after each test."""
    orig_pred = rot_predictor.PreprocessOnlyLagFeatures
    orig_pre = rot_preprocess.PreprocessOnlyLagFeatures
    yield
    rot_predictor.PreprocessOnlyLagFeatures = orig_pred
    rot_preprocess.PreprocessOnlyLagFeatures = orig_pre


def test_preprocess_no_stats_make_features_length_and_fdr_value():
    """Per-horizon feat_dynamic_real: one scalar per channel at t+h."""
    context = 5
    horizon = 4
    preprocess = PreprocessOnlyLagFeaturesNoStats(
        context_window_size=context,
        forecast_horizon=horizon,
        use_feat_static_real=False,
        use_past_feat_dynamic_real=True,
        use_feat_dynamic_real=True,
        use_feat_dynamic_cat=False,
        cardinality=[],
        one_hot_encode=False,
        subtract_mean=False,
        count_nans=False,
    )
    assert preprocess.dynamic_length == context
    assert preprocess.slices_feat_dynamic_real_per_horizon is True

    rng = np.random.default_rng(0)
    t_len = 80
    y = rng.normal(size=t_len).astype(np.float32)
    train_target = y[:-horizon]
    past_row = y[:-horizon]
    fd_row = y

    ts = {
        "target": train_target.tolist(),
        "feat_static_real": [],
        "past_feat_dynamic_real": [past_row.tolist()],
        "feat_dynamic_real": [fd_row.tolist()],
        "feat_dynamic_cat": [],
    }
    legacy_fdr_width = context + horizon  # old no-stats flat FDR block width
    for h in range(horizon):
        feats = preprocess.make_features(ts, starting_index=0, horizon_index=h)
        expected = context + context + 1
        assert len(feats) == expected
        assert len(feats) < context + context + legacy_fdr_width
        # feat order: lags | past | fdr (one float)
        fdr_scalar = feats[-1]
        assert fdr_scalar == pytest.approx(float(fd_row[context + h]))


def test_preprocess_no_stats_apply_patch_trains(
    restore_preprocess_only_lag_features,
):
    apply_rotbaum_no_stats_preprocess_patch()
    assert (
        rot_predictor.PreprocessOnlyLagFeatures
        is PreprocessOnlyLagFeaturesNoStats
    )
    assert (
        rot_preprocess.PreprocessOnlyLagFeatures
        is PreprocessOnlyLagFeaturesNoStats
    )

    horizon = 4
    context = 5
    rng = np.random.default_rng(1)
    t_len = 80
    y = rng.normal(size=t_len).astype(np.float32)

    train = {
        FieldName.TARGET: y[:-horizon].tolist(),
        FieldName.START: pd.Period("2020-01-06", freq="W-MON"),
        FieldName.FEAT_DYNAMIC_REAL: np.vstack([y]).tolist(),
    }
    train["past_feat_dynamic_real"] = (
        np.asarray(train["feat_dynamic_real"], dtype=np.float32)[:, :-horizon]
    ).tolist()

    ds = ListDataset([train], freq="W-MON")

    estimator = TreeEstimator(
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
    predictor = estimator.train(training_data=ds)
    row = predictor.preprocess_object.make_features(
        train, starting_index=0, horizon_index=0
    )
    assert len(row) == context + context + 1
    forecasts = list(predictor.predict(ds))
    assert len(forecasts) == 1
    assert forecasts[0].quantile(0.5).shape == (horizon,)


def test_preprocess_no_stats_subclass_is_stock_preprocessor():
    assert PreprocessOnlyLagFeaturesNoStats.__bases__ == (
        PreprocessOnlyLagFeatures,
    )


def test_use_parallel_preprocess_disabled_with_active_spark_context(
    monkeypatch,
):
    from gluonts.ext.rotbaum._preprocess_no_stats import (
        _use_parallel_preprocess,
    )

    class _FakeSparkContext:
        _active_spark_context = object()

    monkeypatch.setitem(
        __import__("sys").modules,
        "pyspark",
        type("pyspark", (), {"SparkContext": _FakeSparkContext}),
    )
    assert _use_parallel_preprocess(10) is False


def test_use_parallel_preprocess_honours_env_override(monkeypatch):
    from gluonts.ext.rotbaum._preprocess_no_stats import (
        _use_parallel_preprocess,
    )

    monkeypatch.delenv("GLUONTS_ROTBAUM_PREPROCESS_WORKERS", raising=False)
    monkeypatch.setenv("GLUONTS_ROTBAUM_PREPROCESS_WORKERS", "0")
    assert _use_parallel_preprocess(10) is False
    monkeypatch.setenv("GLUONTS_ROTBAUM_PREPROCESS_WORKERS", "2")
    assert _use_parallel_preprocess(1) is True
