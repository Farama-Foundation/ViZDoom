"""
BenchMARL CNN + notification (text) encoder:
+ Notifications are AL (global L)
+ Integer ASCII tokens 1-127, right-padded with zero.
+ Any positive buffer length is accepted, including at runtime.
+ 1 layer GRU, 64 features in hidden state, 32-dim embedding
"""

from dataclasses import dataclass

import torch
from benchmarl.models.cnn import CnnConfig, _number_conv_outputs
from benchmarl.models.common import Model
from torch import nn
from torchrl.modules import GRU, MLP, ConvNet, MultiAgentConvNet, MultiAgentMLP


__all__ = ["TextVisualCnn", "TextVisualCnnConfig"]

_TOKEN_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


class _TextEncoder(nn.Module):
    """Encode each right-padded message independently, with no recurrent memory."""

    def __init__(self, embedding_dim, hidden_size, device):
        super().__init__()
        self.embedding = nn.Embedding(128, embedding_dim, padding_idx=0, device=device)
        self.gru = GRU(embedding_dim, hidden_size, batch_first=True, device=device)

    def forward(self, tokens):
        flat = tokens.reshape(-1, tokens.shape[-1]).long()
        lengths = flat.ne(0).sum(-1)
        embedded = self.embedding(flat)
        outputs, _ = self.gru(embedded)
        last = (lengths - 1).clamp_min(0)[:, None, None]
        features = outputs.gather(1, last.expand(-1, 1, self.gru.hidden_size))[:, 0]
        features = features * lengths.ne(0).unsqueeze(-1)
        return features.reshape(*tokens.shape[:-1], self.gru.hidden_size)


class TextVisualCnn(Model):
    """
    Visual CNN and embedding/GRU text, then a joint MLP.
    + Text encodes messages separately then concatenates in agent order (for centralized CNN).
    + Optional `features` is scaled float vector (role and channel state) appended to MLP input, in agent order when centralized.
    + No normalization layers like AudioVisualCNN
    + Note: Shared centralized critic doesn't return agent dimension. Unshared critics keep outputs from each agents like BenchMARL Cnn.
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
        embedding_dim = kwargs.pop("text_embedding_dim", 32)
        hidden_size = kwargs.pop("text_hidden_size", 64)
        if any(
            type(size) is not int or size <= 0 for size in (embedding_dim, hidden_size)
        ):
            raise ValueError(
                "Text embedding and hidden sizes must be positive integers"
            )
        if kwargs.get("cnn_norm_class") or kwargs.get("mlp_norm_class"):
            raise ValueError("TextVisualCnn does not support normalization layers")
        unexpected = [k for k in kwargs if not k.startswith(("cnn_", "mlp_"))]
        if unexpected:
            raise TypeError(f"Unexpected TextVisualCnn arguments: {unexpected}")
        cnn_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("cnn_")}
        mlp_kwargs = {k[4:]: v for k, v in kwargs.items() if k.startswith("mlp_")}
        channels = self.input_spec[self.visual_key].shape[-1]
        if self.input_has_agent_dim:
            self.visual_cnn = MultiAgentConvNet(
                in_features=channels,
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **cnn_kwargs,
            )
            example = self.visual_cnn._empty_net
        else:
            self.visual_cnn = nn.ModuleList(
                ConvNet(in_features=channels, device=self.device, **cnn_kwargs)
                for _ in range(1 if self.share_params else self.n_agents)
            )
            example = self.visual_cnn[0]
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
        self.text_encoders = nn.ModuleList(
            _TextEncoder(embedding_dim, hidden_size, self.device)
            for _ in range(1 if self.share_params else self.n_agents)
        )
        text_size = hidden_size
        extra_size = (
            0
            if self.features_key is None
            else self.input_spec[self.features_key].shape[-1]
        )
        if self.centralised and self.input_has_agent_dim:
            text_size *= self.n_agents
            extra_size *= self.n_agents
        feature_size = example.out_features * out_h * out_w + text_size + extra_size
        if self.output_has_agent_dim:
            self.mlp = MultiAgentMLP(
                n_agent_inputs=feature_size,
                n_agent_outputs=self.output_leaf_spec.shape[-1],
                n_agents=self.n_agents,
                centralised=self.centralised,
                share_params=self.share_params,
                device=self.device,
                **mlp_kwargs,
            )
        else:
            self.mlp = MLP(
                in_features=feature_size,
                out_features=self.output_leaf_spec.shape[-1],
                device=self.device,
                **mlp_kwargs,
            )

    def _perform_checks(self):
        super()._perform_checks()
        keys = {key[-1] if isinstance(key, tuple) else key: key for key in self.in_keys}
        if len(self.in_keys) != len(keys) or not {
            "observation",
            "notifications",
        } <= set(keys) <= {"observation", "notifications", "features"}:
            raise ValueError(
                "TextVisualCnn requires separate observation and notifications "
                "keys, plus at most an extra features key"
            )
        self.visual_key, self.text_key = keys["observation"], keys["notifications"]
        self.features_key = keys.get("features")
        visual_spec, text_spec = (
            self.input_spec[self.visual_key],
            self.input_spec[self.text_key],
        )
        shape, text_shape = visual_spec.shape, text_spec.shape
        if len(shape) != (4 if self.input_has_agent_dim else 3) or len(text_shape) != (
            2 if self.input_has_agent_dim else 1
        ):
            raise ValueError("Expected AHWC images and AL text (global HWC and L)")
        if self.n_agents <= 0 or (
            self.input_has_agent_dim
            and (shape[0] != self.n_agents or text_shape[0] != self.n_agents)
        ):
            raise ValueError("Input agent dimensions must match positive n_agents")
        if any(size <= 0 for size in (*shape, *text_shape)):
            raise ValueError("Input dimensions must be positive")
        self._check_dtypes(visual_spec.dtype, text_spec.dtype)
        if self.features_key is not None:
            features_spec = self.input_spec[self.features_key]
            if len(features_spec.shape) != (2 if self.input_has_agent_dim else 1) or (
                self.input_has_agent_dim and features_spec.shape[0] != self.n_agents
            ):
                raise ValueError("Expected AF features (global F)")
            if features_spec.shape[-1] <= 0:
                raise ValueError("Input dimensions must be positive")
            if not features_spec.dtype.is_floating_point:
                raise TypeError("Features must be a scaled floating point vector")
        output_shape = self.output_leaf_spec.shape
        if (
            len(output_shape) != (2 if self.output_has_agent_dim else 1)
            or output_shape[-1] <= 0
        ):
            raise ValueError("Output spec must be (agents, features) or (features,)")
        if self.output_has_agent_dim and output_shape[0] != self.n_agents:
            raise ValueError("Output agent dimension must match n_agents")

    @staticmethod
    def _check_dtypes(image_dtype, text_dtype):
        if image_dtype != torch.uint8 and not image_dtype.is_floating_point:
            raise TypeError(
                "Observation must be uint8 or floating point (already normalized)"
            )
        if text_dtype not in _TOKEN_DTYPES:
            raise TypeError(
                "Notifications must be integer ASCII tokens, not normalized floats"
            )

    def _check_inputs(self, image, tokens):
        self._check_dtypes(image.dtype, tokens.dtype)
        image_shape = self.input_spec[self.visual_key].shape
        if image.shape[-len(image_shape) :] != image_shape:
            raise ValueError("Observation shape must match the image spec")
        if (
            tokens.ndim != image.ndim - 2
            or tokens.shape[:-1] != image.shape[:-3]
            or tokens.shape[-1] <= 0
        ):
            raise ValueError(
                "Notifications shape must match image batch/agent dimensions "
                "with positive length"
            )
        # GAE vmaps over current/next observations
        if not torch._C._functorch.is_batchedtensor(tokens):
            if ((tokens < 0) | (tokens > 127)).any():
                raise ValueError("Notifications tokens must be in 0..127")
            if (tokens[..., :-1].eq(0) & tokens[..., 1:].ne(0)).any():
                raise ValueError("Notifications must be right-padded with zero")

    def _visual_features(self, image):
        image = image.float() / 255 if image.dtype == torch.uint8 else image.float()
        image = image.movedim(-1, -3)
        if self.input_has_agent_dim:
            features = self.visual_cnn(image)
            return features if self.output_has_agent_dim else features[..., 0, :]
        if self.share_params:
            return self.visual_cnn[0](image)
        return torch.stack([net(image) for net in self.visual_cnn], dim=-2)

    def _text_features(self, tokens):
        if self.share_params:
            features = self.text_encoders[0](tokens)
            if self.centralised and self.input_has_agent_dim:
                features = features.flatten(-2)
            return features
        if self.centralised:
            features = [encoder(tokens) for encoder in self.text_encoders]
            if self.input_has_agent_dim:
                features = [feature.flatten(-2) for feature in features]
        else:
            features = [
                encoder(tokens[..., agent, :])
                for agent, encoder in enumerate(self.text_encoders)
            ]
        return torch.stack(features, dim=-2)

    def _extra_features(self, values):
        if not (self.centralised and self.input_has_agent_dim):
            return values
        values = values.flatten(-2)
        if self.output_has_agent_dim:
            return values.unsqueeze(-2).expand(
                *values.shape[:-1], self.n_agents, values.shape[-1]
            )
        return values

    def _forward(self, tensordict):
        image, tokens = tensordict[self.visual_key], tensordict[self.text_key]
        self._check_inputs(image, tokens)
        parts = [self._visual_features(image), self._text_features(tokens)]
        if self.features_key is not None:
            parts.append(self._extra_features(tensordict[self.features_key].float()))
        tensordict.set(self.out_key, self.mlp(torch.cat(parts, dim=-1)))
        return tensordict


@dataclass
class TextVisualCnnConfig(CnnConfig):
    text_embedding_dim: int = 32
    text_hidden_size: int = 64

    @property
    def name(self):
        return "text_visual_cnn"

    @staticmethod
    def associated_class():
        return TextVisualCnn
