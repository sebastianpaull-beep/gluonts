# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import numpy as np

from gluonts.dataset.common import ListDataset
from gluonts.mx import FeedForwardEstimator, Trainer
from gluonts.mx.distribution import StudentTOutput


def test_feedforward_train_with_feat_dynamic_real():
    prediction_length = 3
    context_length = 4
    num_feat = 1
    length = 30
    freq = "D"

    data = [
        {
            "start": "2020-01-01",
            "target": np.random.randn(length).astype("float32"),
            "feat_dynamic_real": np.random.randn(num_feat, length).astype(
                "float32"
            ),
        }
    ]

    ds = ListDataset(data, freq=freq)

    estimator = FeedForwardEstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_feat_dynamic_real=num_feat,
        distr_output=StudentTOutput(),
        sampling=False,
        mean_scaling=False,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1, hybridize=False),
        batch_size=1,
        num_hidden_dimensions=[8, 8],
    )

    predictor = estimator.train(training_data=ds)
    assert predictor is not None


def test_feedforward_num_feat_zero_unchanged():
    prediction_length = 2
    context_length = 3
    length = 20
    freq = "D"

    data = [
        {
            "start": "2020-01-01",
            "target": np.random.randn(length).astype("float32"),
        }
    ]

    ds = ListDataset(data, freq=freq)

    estimator = FeedForwardEstimator(
        prediction_length=prediction_length,
        context_length=context_length,
        num_feat_dynamic_real=0,
        distr_output=StudentTOutput(),
        sampling=False,
        mean_scaling=False,
        trainer=Trainer(epochs=1, num_batches_per_epoch=1, hybridize=False),
        batch_size=1,
        num_hidden_dimensions=[8, 8],
    )

    predictor = estimator.train(training_data=ds)
    assert predictor is not None
