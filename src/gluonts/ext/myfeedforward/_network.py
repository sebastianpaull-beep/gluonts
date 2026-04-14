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

from typing import Optional

import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor


class MyFeedForwardNetworkBase(mx.gluon.HybridBlock):
    """Point FeedForward with optional flattened dynamic real features."""

    @validated()
    def __init__(
        self,
        prediction_length: int,
        num_cells: int,
        context_length: int,
        num_feat_dynamic_real: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        assert num_feat_dynamic_real >= 0
        self.prediction_length = prediction_length
        self.num_cells = num_cells
        self.context_length = context_length
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.flat_dim = context_length + num_feat_dynamic_real * (
            context_length + prediction_length
        )

        with self.name_scope():
            self.nn = mx.gluon.nn.HybridSequential()
            self.nn.add(
                mx.gluon.nn.Dense(units=self.num_cells, activation="relu")
            )
            self.nn.add(
                mx.gluon.nn.Dense(units=self.num_cells, activation="relu")
            )
            self.nn.add(
                mx.gluon.nn.Dense(
                    units=self.prediction_length, activation="softrelu"
                )
            )

    def _mlp_input(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Optional[Tensor],
        future_feat_dynamic_real: Optional[Tensor],
    ) -> Tensor:
        if self.num_feat_dynamic_real == 0:
            return past_target
        assert past_feat_dynamic_real is not None
        assert future_feat_dynamic_real is not None
        st = F.reshape(past_target, (0, -1))
        pf = F.reshape(past_feat_dynamic_real, (0, -1))
        ff = F.reshape(future_feat_dynamic_real, (0, -1))
        return F.concat(st, pf, ff, dim=1)


class MyFeedForwardTrainNetwork(MyFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target,
        future_target,
        past_feat_dynamic_real=None,
        future_feat_dynamic_real=None,
    ):
        mlp_in = self._mlp_input(
            F, past_target, past_feat_dynamic_real, future_feat_dynamic_real
        )
        prediction = self.nn(mlp_in)
        return (prediction - future_target).abs().mean(axis=-1)


class MyFeedForwardPredNetwork(MyFeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target,
        past_feat_dynamic_real=None,
        future_feat_dynamic_real=None,
    ):
        mlp_in = self._mlp_input(
            F, past_target, past_feat_dynamic_real, future_feat_dynamic_real
        )
        prediction = self.nn(mlp_in)
        return prediction.expand_dims(axis=1)
