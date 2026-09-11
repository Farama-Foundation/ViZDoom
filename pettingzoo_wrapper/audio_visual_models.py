"""
Use ``AudioVisualCnnConfig`` with the same ``cnn_*`` and ``mlp_*`` arguments as BenchMARL's ``CnnConfig``.

CNN settings: [32, 64, 64] channels, [8, 4, 3] kernels, [4, 2, 1] strides, zero padding, ReLU, and a [512] ReLU MLP. Each branch receives all frames, with both stereo channels in the audio branch.

Note: Byte observations are divided by 255 once, so floating inputs must already be normalized.
"""

from dataclasses import dataclass

import torch
from benchmarl.models.cnn import CnnConfig, _number_conv_outputs
from benchmarl.models.common import Model
from torch import nn
from torchrl.modules import MLP, ConvNet, MultiAgentConvNet, MultiAgentMLP


__all__ = ["AudioVisualCnn", "AudioVisualCnnConfig"]


class AudioVisualCnn(Model):
    """
    Independent visual/audio CNNs, flattened feature concat, then one MLP. Same as BenchMARL Cnn: decentralized agents don't mix observations, centralized branches pool agents as channels.
    + Shared centralized critic returns one output without an agent dimension.
    + Unshared centralized critic keeps per agent outputs and its MLP pools the per agent branch features, as in the original Cnn.
    + Global HWC image input is supported for centralized models: inputs are separate HWC ``observation`` (RGB, 3 channels per frame) and ``audio`` (stereo STFT, 2 channels per frame) fields.
    """

    def __init__(self, **kwargs):
        super().__init__(
            **{
                key: kwargs.pop(key)
                for key in (
                    "input_spec",
                    "output_spec",
                    "agent_group",
                    "input_has_agent_dim",
                    "n_agents",
                    "centralised",
                    "share_params",
                    "device",
                    "action_spec",
                    "model_index",
                    "is_critic",
                )
            }
        )
        if kwargs.get("cnn_norm_class") or kwargs.get("mlp_norm_class"):
            raise ValueError("AudioVisualCnn does not support normalization layers")
        unexpected = [k for k in kwargs if not k.startswith(("cnn_", "mlp_"))]
        if unexpected:
            raise TypeError(f"Unexpected AudioVisualCnn arguments: {unexpected}")
        cnn_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("cnn_")}
        mlp_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("mlp_")}
        self.visual_cnn = self._make_branch(
            self.input_spec[self.visual_key].shape[-1], cnn_kwargs
        )
        self.audio_cnn = self._make_branch(
            self.input_spec[self.audio_key].shape[-1], cnn_kwargs
        )
        example = (
            self.visual_cnn._empty_net
            if self.input_has_agent_dim
            else self.visual_cnn[0]
        )
        out_h, out_w = _number_conv_outputs(
            self.input_spec[self.visual_key].shape[-3:-1],
            example.paddings,
            example.kernel_sizes,
            example.strides,
        )
        if out_h <= 0 or out_w <= 0:
            raise ValueError(
                "Observation spatial size is too small for the CNN kernels"
            )
        self.branch_output_size = example.out_features * out_h * out_w
        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=2 * self.branch_output_size,
                n_agent_outputs=self.output_leaf_spec.shape[-1],
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **mlp_kwargs,
            )
        else:
            self.mlp = MLP(
                in_features=2 * self.branch_output_size,
                out_features=self.output_leaf_spec.shape[-1],
                device=self.device,
                **mlp_kwargs,
            )

    def _perform_checks(self):
        super()._perform_checks()
        keys = {key[-1] if isinstance(key, tuple) else key: key for key in self.in_keys}
        if len(self.in_keys) != 2 or set(keys) != {"observation", "audio"}:
            raise ValueError(
                "AudioVisualCnn requires separate observation and audio keys"
            )
        self.visual_key, self.audio_key = keys["observation"], keys["audio"]
        shape = self.input_spec[self.visual_key].shape
        audio_shape = self.input_spec[self.audio_key].shape
        expected_rank = 4 if self.input_has_agent_dim else 3
        if len(shape) != expected_rank or audio_shape[:-1] != shape[:-1]:
            raise ValueError(
                "AudioVisualCnn expects matching AHWC (or global HWC) image specs"
            )
        if self.input_has_agent_dim and shape[0] != self.n_agents:
            raise ValueError("Observation agent dimension must match n_agents")
        if any(size <= 0 for size in shape):
            raise ValueError("Observation dimensions must be positive")
        if shape[-1] % 3 or audio_shape[-1] != 2 * (shape[-1] // 3):
            raise ValueError("Expected 3 RGB and 2 audio channels per stacked frame")
        output_shape = self.output_leaf_spec.shape
        expected_output_rank = 2 if self.output_has_agent_dim else 1
        if len(output_shape) != expected_output_rank or output_shape[-1] <= 0:
            raise ValueError("Output spec must be (agents, features) or (features,)")
        if self.output_has_agent_dim and output_shape[-2] != self.n_agents:
            raise ValueError("Output agent dimension must match n_agents")

    def _make_branch(self, channels, cnn_kwargs):
        if self.input_has_agent_dim:
            return MultiAgentConvNet(
                in_features=channels,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **cnn_kwargs,
            )
        return nn.ModuleList(
            ConvNet(in_features=channels, device=self.device, **cnn_kwargs)
            for _ in range(1 if self.share_params else self.n_agents)
        )

    def _branch_features(self, branch, image):
        if image.dtype == torch.uint8:
            image = image.float() / 255
        else:
            image = image.float()
        image = image.movedim(-1, -3)
        if self.input_has_agent_dim:
            features = branch(image)
            return features if self.output_has_agent_dim else features[..., 0, :]
        if self.share_params:
            return branch[0](image)
        return torch.stack([net(image) for net in branch], dim=-2)

    def _forward(self, tensordict):
        features = torch.cat(
            (
                self._branch_features(self.visual_cnn, tensordict[self.visual_key]),
                self._branch_features(self.audio_cnn, tensordict[self.audio_key]),
            ),
            dim=-1,
        )
        tensordict.set(self.out_key, self.mlp(features))
        return tensordict


@dataclass
class AudioVisualCnnConfig(CnnConfig):
    """Separate RGB/stereo-STFT encoders; channel counts come from input specs."""

    @property
    def name(self):
        return "audio_visual_cnn"

    @staticmethod
    def associated_class():
        return AudioVisualCnn
