"""
Use ``AudioVisualCnnConfig`` with the same ``cnn_*`` and ``mlp_*`` arguments as
BenchMARL's ``CnnConfig``. Nature CNN settings are [32, 64, 64] channels,
[8, 4, 3] kernels, [4, 2, 1] strides, zero padding, ReLU, and a [512] ReLU MLP.
Each branch receives all frames, with both stereo channels in the audio branch.
There is no recurrence, attention, raw waveform processing, or normalization
layer. Byte observations are divided by 255 once; floating inputs must already
be normalized. Neither the input tensor nor its TensorDict entry is replaced.
"""

from dataclasses import dataclass
from typing import Optional

import torch
from benchmarl.models.cnn import CnnConfig, _number_conv_outputs
from benchmarl.models.common import Model
from torch import nn
from torchrl.modules import MLP, ConvNet, MultiAgentConvNet, MultiAgentMLP


__all__ = ["AudioVisualCnn", "AudioVisualCnnConfig", "split_audio_visual_observation"]


def split_audio_visual_observation(observation, frame_stack=None):
    """Split ``(*batch, [agents,] H, W, 5 * frames)`` into CHW modalities.

    The repeated per-frame order is ``[R, G, B, L_STFT, R_STFT]``, not a
    contiguous RGB block followed by audio. Floating tensors are assumed to be
    normalized already (no value-dependent scaling or per-frame normalization).
    Returned tensors have fresh storage, including for floating-point inputs.
    """
    if not isinstance(observation, torch.Tensor):
        raise TypeError("Packed observations must be torch tensors")
    if observation.ndim < 3:
        raise ValueError("Packed observations must end in H, W, channels")
    channels = observation.shape[-1]
    if channels == 0 or channels % 5:
        raise ValueError(
            "Packed channels must be 5 * frame_stack (RGB, L_STFT, R_STFT)"
        )
    if frame_stack is not None and channels != 5 * frame_stack:
        raise ValueError("Packed channels do not match frame_stack")
    if observation.dtype == torch.uint8:
        image = observation.to(torch.float32) / 255
    elif observation.is_floating_point():
        image = observation.to(torch.float32)
    else:
        raise TypeError(
            "Packed observations must be uint8 or normalized floating point"
        )
    frames = image.reshape(*image.shape[:-1], channels // 5, 5)
    visual = frames[..., :3].flatten(-2).movedim(-1, -3).contiguous()
    audio = frames[..., 3:].flatten(-2).movedim(-1, -3).contiguous()
    return visual, audio


class AudioVisualCnn(Model):
    """Independent visual/audio CNNs, flattened feature concat, then one MLP.

    Agent sharing and centralized pooling follow BenchMARL's Cnn: decentralized
    agents never mix observations; centralized branches pool agents as channels.
    A shared centralized critic returns one output without an agent dimension.
    An unshared centralized critic retains per-agent outputs and its MLP pools
    the per-agent branch features, as in the original Cnn. Global HWC image input
    is supported for centralized models. Multiple observation keys and vector
    observations are deliberately rejected rather than silently concatenated.
    """

    def __init__(self, frame_stack=None, **kwargs):
        self.frame_stack = frame_stack
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
        self.visual_cnn = self._make_branch(3 * self.frame_stack, cnn_kwargs)
        self.audio_cnn = self._make_branch(2 * self.frame_stack, cnn_kwargs)
        example = (
            self.visual_cnn._empty_net
            if self.input_has_agent_dim
            else self.visual_cnn[0]
        )
        out_h, out_w = _number_conv_outputs(
            self.input_leaf_spec.shape[-3:-1],
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
        if len(self.in_keys) != 1:
            raise ValueError(
                "AudioVisualCnn requires exactly one packed observation key"
            )
        shape = self.input_leaf_spec.shape
        expected_rank = 4 if self.input_has_agent_dim else 3
        if len(shape) != expected_rank:
            raise ValueError(
                "AudioVisualCnn expects an AHWC (or global HWC) image spec"
            )
        if self.input_has_agent_dim and shape[0] != self.n_agents:
            raise ValueError("Observation agent dimension must match n_agents")
        if any(size <= 0 for size in shape):
            raise ValueError("Observation dimensions must be positive")
        if shape[-1] % 5:
            raise ValueError(
                "Packed channels must be 5 * frame_stack (RGB, L_STFT, R_STFT)"
            )
        if self.frame_stack is None:
            self.frame_stack = shape[-1] // 5
        if (
            isinstance(self.frame_stack, bool)
            or not isinstance(self.frame_stack, int)
            or self.frame_stack <= 0
            or shape[-1] != 5 * self.frame_stack
        ):
            raise ValueError(
                "frame_stack must be positive and match the packed channels"
            )
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
        if self.input_has_agent_dim:
            features = branch(image)
            return features if self.output_has_agent_dim else features[..., 0, :]
        if self.share_params:
            return branch[0](image)
        return torch.stack([net(image) for net in branch], dim=-2)

    def _forward(self, tensordict):
        observation = tensordict.get(self.in_key)
        if not isinstance(observation, torch.Tensor):
            raise TypeError("Packed observations must be torch tensors")
        shape = self.input_leaf_spec.shape
        if observation.shape[-len(shape) :] != shape:
            raise ValueError(f"Expected observation trailing shape {tuple(shape)}")
        visual, audio = split_audio_visual_observation(observation, self.frame_stack)
        features = torch.cat(
            (
                self._branch_features(self.visual_cnn, visual),
                self._branch_features(self.audio_cnn, audio),
            ),
            dim=-1,
        )
        tensordict.set(self.out_key, self.mlp(features))
        return tensordict


@dataclass
class AudioVisualCnnConfig(CnnConfig):
    """Drop-in CnnConfig with a separate encoder for each modality.

    ``frame_stack=None`` infers the stack from the single input spec; set an
    integer to enforce an exact stack size. Twenty packed channels means four
    frames. ``name`` is a string property, accessed as ``config.name``.
    """

    frame_stack: Optional[int] = None

    @property
    def name(self):
        return "audio_visual_cnn"

    @staticmethod
    def associated_class():
        return AudioVisualCnn
