from typing import Optional, Text

import model_lib
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.specs import distribution_spec, tensor_spec
from tf_agents.typing import types
from tf_agents.utils import nest_utils


class GRUModel(network.Network):
    def __init__(self, input_tensor_spec, output_tensor_spec, name=None):
        super(GRUModel, self).__init__(
            input_tensor_spec=input_tensor_spec, state_spec=(), name=name
        )

        self.model = model_lib.GRUTrainingModel()

    def call(self, inputs, network_state=()):
        policy_logits, value = self._model(inputs)
        return {"logits": policy_logits, "value": value}, network_state


class GRUPolicyModel(network.DistributionNetwork):
    def __init__(
        self,
        shared_network: network.Network,
        input_tensors_spec: types.NestedTensorSpec,
        output_tensors_spec,
        name="GRUPolicyModel",
    ):
        super(GRUPolicyModel, self).__init__(
            input_tensor_spec=input_tensors_spec,
            state_spec=(),
            output_spec=output_tensors_spec,
            name=name,
        )

        self._input_tensors_spec = input_tensors_spec
        self._shared_network = shared_network
        self._output_tensors_spec = output_tensors_spec

        n_unique_actions = np.unique(
            output_tensors_spec.maximum - output_tensors_spec.minimum + 1
        )
        input_param_spec = {
            "logits": tensor_spec.TensorSpec(
                shape=n_unique_actions, dtype=tf.float32, name=name + "_logits"
            )  # type: ignore
        }
        self._output_dist_spec = distribution_spec.DistributionSpec(
            tfp.distributions.Categorical,
            input_param_spec,
            sample_spec=output_tensors_spec,
            dtype=output_tensors_spec.dtype,
        )

    def call(self, inputs, network_state=()):
        outer_rank = nest_utils.get_outer_rank(inputs, self._input_tensors_spec)
        if outer_rank == 0:
            inputs = tf.nest.map_structure(lambda x: tf.reshape(x, (1, -1)), inputs)
        model_out, _ = self._shared_network(inputs)

        output_dist = self._output_dist_spec.build_distribution(logits=model_out)

        return output_dist, network_state


class GRUValueModel(network.Network):
    """Circuit GRL Model."""

    def __init__(
        self,
        input_tensors_spec: types.NestedTensorSpec,
        shared_network: network.Network,
        name: Optional[Text] = None,
    ):
        super(GRUValueModel, self).__init__(
            input_tensor_spec=input_tensors_spec, state_spec=(), name=name
        )

        self._input_tensors_spec = input_tensors_spec
        self._shared_network = shared_network

    def call(self, inputs, network_state=()):
        outer_rank = nest_utils.get_outer_rank(inputs, self._input_tensors_spec)
        if outer_rank == 0:
            inputs = tf.nest.map_structure(lambda x: tf.reshape(x, (1, -1)), inputs)
        model_out, _ = self._shared_network(
            inputs, finetune_value_only=self._finetune_value_only
        )

        def squeeze_value_dim(value):
            # Make value_prediction's shape from [B, T, 1] to [B, T].
            return tf.squeeze(value, -1)

        return squeeze_value_dim(model_out["value"]), network_state


def create_grl_models(
    observation_tensor_spec: types.NestedTensorSpec,
    action_tensor_spec: types.NestedTensorSpec,
):
    grl_shared_net = GRUModel(
        observation_tensor_spec,
        action_tensor_spec,
    )
    grl_actor_net = GRUPolicyModel(
        grl_shared_net, observation_tensor_spec, action_tensor_spec
    )
    grl_value_net = GRUValueModel(observation_tensor_spec, grl_shared_net)
    return grl_actor_net, grl_value_net
