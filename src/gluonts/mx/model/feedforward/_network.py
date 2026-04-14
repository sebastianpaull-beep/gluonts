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

from typing import List, Optional, Tuple

import mxnet as mx

from gluonts.core.component import validated
from gluonts.mx import Tensor
from gluonts.mx.block.scaler import MeanScaler, NOPScaler
from gluonts.mx.distribution import DistributionOutput
from gluonts.mx.util import weighted_average


class FeedForwardNetworkBase(mx.gluon.HybridBlock):
    """
    Abstract base class to implement feed-forward networks for probabilistic
    time series prediction.

    This class does not implement hybrid_forward: this is delegated
    to the two subclasses FeedForwardTrainingNetwork and
    FeedForwardPredictionNetwork, that define respectively how to
    compute the loss and how to generate predictions.

    Parameters
    ----------
    num_hidden_dimensions
        Number of hidden nodes in each layer.
    prediction_length
        Number of time units to predict.
    context_length
        Number of time units that condition the predictions.
    batch_normalization
        Whether to use batch normalization.
    mean_scaling
        Scale the network input by the data mean and the network output by
        its inverse.
    distr_output
        Distribution to fit.
    num_feat_dynamic_real
        Number of dynamic real feature channels (past and future windows).
        When 0, only ``past_target`` is used (original behaviour). When positive,
        ``past_feat_dynamic_real`` and ``future_feat_dynamic_real`` are flattened
        and concatenated with the scaled past target before the MLP.
    kwargs
    """

    # Needs the validated decorator so that arguments types are checked and
    # the block can be serialized.
    @validated()
    def __init__(
        self,
        num_hidden_dimensions: List[int],
        prediction_length: int,
        context_length: int,
        batch_normalization: bool,
        mean_scaling: bool,
        distr_output: DistributionOutput,
        num_feat_dynamic_real: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        assert num_feat_dynamic_real >= 0
        self.num_hidden_dimensions = num_hidden_dimensions
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_normalization = batch_normalization
        self.mean_scaling = mean_scaling
        self.distr_output = distr_output
        self.num_feat_dynamic_real = num_feat_dynamic_real
        self.flat_dim = context_length + num_feat_dynamic_real * (
            context_length + prediction_length
        )

        with self.name_scope():
            self.distr_args_proj = self.distr_output.get_args_proj()
            self.mlp = mx.gluon.nn.HybridSequential()
            dims = self.num_hidden_dimensions
            for layer_no, units in enumerate(dims[:-1]):
                self.mlp.add(mx.gluon.nn.Dense(units=units, activation="relu"))
                if self.batch_normalization:
                    self.mlp.add(mx.gluon.nn.BatchNorm())
            self.mlp.add(mx.gluon.nn.Dense(units=prediction_length * dims[-1]))
            self.mlp.add(
                mx.gluon.nn.HybridLambda(
                    lambda F, o: F.reshape(
                        o, (-1, prediction_length, dims[-1])
                    )
                )
            )
            self.scaler = MeanScaler() if mean_scaling else NOPScaler()

    def _mlp_input_from_scaled(
        self,
        F,
        scaled_target: Tensor,
        past_feat_dynamic_real: Optional[Tensor],
        future_feat_dynamic_real: Optional[Tensor],
    ) -> Tensor:
        """
        Build a flat MLP input: [flatten(scaled past target), flatten(past
        features), flatten(future features)] when ``num_feat_dynamic_real > 0``.
        """
        if self.num_feat_dynamic_real == 0:
            return scaled_target
        assert past_feat_dynamic_real is not None
        assert future_feat_dynamic_real is not None
        st = F.reshape(scaled_target, (0, -1))
        pf = F.reshape(past_feat_dynamic_real, (0, -1))
        ff = F.reshape(future_feat_dynamic_real, (0, -1))
        return F.concat(st, pf, ff, dim=1)

    def get_distr_args(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Optional[Tensor] = None,
        future_feat_dynamic_real: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Given past target values (and optional dynamic real features), applies
        the feed-forward network and maps the output to distribution parameters.

        Parameters
        ----------
        F
        past_target
            Tensor containing past target observations.
            Shape: (batch_size, context_length) for univariate.
        past_feat_dynamic_real
            Optional. Shape: (batch_size, F, context_length) when
            ``num_feat_dynamic_real > 0``.
        future_feat_dynamic_real
            Optional. Shape: (batch_size, F, prediction_length) when
            ``num_feat_dynamic_real > 0``.

        Returns
        -------
        Tensor
            The parameters of distribution.
        Tensor
            An array containing the location (shift) of the distribution.
        Tensor
            An array containing the scale of the distribution.
        """
        scaled_target, target_scale = self.scaler(
            past_target,
            F.ones_like(past_target),
        )
        mlp_in = self._mlp_input_from_scaled(
            F,
            scaled_target,
            past_feat_dynamic_real,
            future_feat_dynamic_real,
        )
        mlp_outputs = self.mlp(mlp_in)
        distr_args = self.distr_args_proj(mlp_outputs)
        scale = target_scale.expand_dims(axis=1)
        loc = F.zeros_like(scale)
        return distr_args, loc, scale


class FeedForwardTrainingNetwork(FeedForwardNetworkBase):
    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        future_target: Tensor,
        future_observed_values: Tensor,
        past_feat_dynamic_real: Optional[Tensor] = None,
        future_feat_dynamic_real: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes a probability distribution for future data given the past, and
        returns the loss associated with the actual future observations.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Tensor with future observations.
            Shape: (batch_size, prediction_length, target_dim).
        future_observed_values
            Tensor indicating which values in the target are observed, and
            which ones are imputed instead.
        past_feat_dynamic_real
            Optional past window of dynamic real features when
            ``num_feat_dynamic_real > 0``.
        future_feat_dynamic_real
            Optional future window of dynamic real features when
            ``num_feat_dynamic_real > 0``.

        Returns
        -------
        Tensor
            Loss tensor. Shape: (batch_size, ).
        """
        distr_args, loc, scale = self.get_distr_args(
            F,
            past_target,
            past_feat_dynamic_real,
            future_feat_dynamic_real,
        )
        distr = self.distr_output.distribution(
            distr_args, loc=loc, scale=scale
        )

        # (batch_size, prediction_length, target_dim)
        loss = distr.loss(future_target)

        weighted_loss = weighted_average(
            F=F, x=loss, weights=future_observed_values, axis=1
        )

        # (batch_size, )
        return weighted_loss


class FeedForwardSamplingNetwork(FeedForwardNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Optional[Tensor] = None,
        future_feat_dynamic_real: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes a probability distribution for future data given the past, and
        draws samples from it.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        past_feat_dynamic_real
            Optional past window of dynamic real features when
            ``num_feat_dynamic_real > 0``.
        future_feat_dynamic_real
            Optional future window of dynamic real features when
            ``num_feat_dynamic_real > 0``.

        Returns
        -------
        Tensor
            Prediction sample. Shape: (batch_size, samples, prediction_length).
        """

        distr_args, loc, scale = self.get_distr_args(
            F,
            past_target,
            past_feat_dynamic_real,
            future_feat_dynamic_real,
        )
        distr = self.distr_output.distribution(
            distr_args, loc=loc, scale=scale
        )

        # (num_samples, batch_size, prediction_length)
        samples = distr.sample(self.num_parallel_samples)

        # (batch_size, num_samples, prediction_length)
        return samples.swapaxes(0, 1)


class FeedForwardDistributionNetwork(FeedForwardNetworkBase):
    @validated()
    def __init__(
        self, num_parallel_samples: int = 100, *args, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.num_parallel_samples = num_parallel_samples

    def hybrid_forward(
        self,
        F,
        past_target: Tensor,
        past_feat_dynamic_real: Optional[Tensor] = None,
        future_feat_dynamic_real: Optional[Tensor] = None,
    ) -> Tensor:
        """
        Computes the parameters of distribution for future data given the past,
        and draws samples from it.

        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        past_feat_dynamic_real
            Optional past window of dynamic real features when
            ``num_feat_dynamic_real > 0``.
        future_feat_dynamic_real
            Optional future window of dynamic real features when
            ``num_feat_dynamic_real > 0``.

        Returns
        -------
        Tensor
            The parameters of distribution.
        Tensor
            An array containing the location (shift) of the distribution.
        Tensor
            An array containing the scale of the distribution.
        """
        distr_args, loc, scale = self.get_distr_args(
            F,
            past_target,
            past_feat_dynamic_real,
            future_feat_dynamic_real,
        )
        return distr_args, loc, scale
