"""
Reference: https://github.com/minqi/learning-to-communicate-pytorch/
paper: https://arxiv.org/pdf/1605.06676

Factorized movement/message PPO for the discrete chat channel.
+ The environment action is [motor, message], and sum log prob is: log pi(a, m | h) = log pi_move(a | h) + log pi_msg(m | h), and each head has separate entropy coefficient.
+ Role masks force the informer motor head to no-op and the operator message head to silence
+ Message logits are also masked whenever the engine cannot transmit (send cooldown, or communication disabled), so PPO dont credit tokens that were never said.
+ Gradients stop at the sampled token exactly as in RIAL.
"""

from dataclasses import MISSING, dataclass

import torch
from benchmarl.algorithms import Ippo, IppoConfig, Mappo, MappoConfig
from benchmarl.algorithms.common import Algorithm
from benchmarl.models.common import ModelConfig
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import distributions as d
from torchrl.data import Composite, Unbounded
from torchrl.modules import ProbabilisticActor
from torchrl.objectives import ClipPPOLoss, ValueEstimators

from .communication import (
    FEATURE_ROLE_INFORMER,
    FEATURE_ROLE_OPERATOR,
    FEATURE_SEND_AVAILABLE,
)


__all__ = [
    "FactorizedCategorical",
    "FactorizedClipPPOLoss",
    "FactorizedIppo",
    "FactorizedIppoConfig",
    "FactorizedMappo",
    "FactorizedMappoConfig",
    "mask_factorized_logits",
    "mask_message_logits",
]


class FactorizedCategorical(d.Distribution):
    """Independent categoricals over nvec heads sharing one logit vector"""

    arg_constraints = {}
    has_enumerate_support = False

    def __init__(self, logits, nvec, validate_args=None):
        nvec = tuple(int(size) for size in nvec)
        if len(nvec) < 2 or any(size < 1 for size in nvec):
            raise ValueError("nvec must hold at least two positive head sizes")
        if logits.shape[-1] != sum(nvec):
            raise ValueError("logits must concatenate exactly the per-head logits")
        self.nvec = nvec
        self.dists = [
            d.Categorical(logits=chunk)
            for chunk in logits.split(nvec, dim=-1)  # type: ignore[arg-type]
        ]
        super().__init__(
            batch_shape=logits.shape[:-1],
            event_shape=torch.Size([len(nvec)]),
            validate_args=False,
        )

    @property
    def mode(self):
        return torch.stack([dist.logits.argmax(-1) for dist in self.dists], dim=-1)

    @property
    def deterministic_sample(self):
        return self.mode

    def sample(self, sample_shape=torch.Size()):  # noqa: B008
        with torch.no_grad():
            return torch.stack(
                [dist.sample(sample_shape) for dist in self.dists], dim=-1
            )

    def rsample(self, sample_shape=torch.Size()):  # noqa: B008
        raise NotImplementedError("A discrete channel is not reparameterizable")

    def log_prob(self, value):
        value = value.long()
        if value.shape[-1] != len(self.dists):
            raise ValueError("value must hold one index per head")
        return sum(
            dist.log_prob(value[..., head]) for head, dist in enumerate(self.dists)
        )

    def entropy_heads(self):
        return [dist.entropy() for dist in self.dists]

    def entropy(self):
        return sum(self.entropy_heads())


def mask_factorized_logits(
    logits, features, n_motor, n_tokens, informer_can_move: bool = False
):
    """Apply role and channel constraints to factorized action logits"""
    if logits.shape[-1] != n_motor + n_tokens:
        raise ValueError("logits must hold the movement then message heads")
    if features.shape[-1] <= FEATURE_SEND_AVAILABLE:
        raise ValueError("features must hold role and channel state")

    motor = logits[..., :n_motor]
    message = logits[..., n_motor:]
    non_idle = torch.ones_like(motor, dtype=torch.bool)
    non_idle[..., 0] = False
    non_silence = torch.ones_like(message, dtype=torch.bool)
    non_silence[..., 0] = False

    informer = features[..., FEATURE_ROLE_INFORMER] > 0.5
    if informer_can_move:
        informer = torch.zeros_like(informer)
    operator = features[..., FEATURE_ROLE_OPERATOR] > 0.5
    send_blocked = features[..., FEATURE_SEND_AVAILABLE] <= 0.5
    return torch.cat(
        (
            motor.masked_fill(
                informer.unsqueeze(-1) & non_idle,
                torch.finfo(motor.dtype).min,
            ),
            message.masked_fill(
                (operator | send_blocked).unsqueeze(-1) & non_silence,
                torch.finfo(message.dtype).min,
            ),
        ),
        dim=-1,
    )


mask_message_logits = mask_factorized_logits


class FactorizedClipPPOLoss(ClipPPOLoss):
    """Clipped PPO with one entropy coefficient per action head"""

    def __init__(self, *args, head_entropy_coeffs, **kwargs):
        super().__init__(*args, entropy_coeff=1.0, **kwargs)
        coeffs = [float(coeff) for coeff in head_entropy_coeffs]
        if any(coeff < 0 for coeff in coeffs):
            raise ValueError("Entropy coefficients must be nonnegative")
        self.head_entropy_coeffs = coeffs
        self._head_entropies = None

    def _get_entropy(self, dist, adv_shape):
        if not isinstance(dist, FactorizedCategorical):
            return super()._get_entropy(dist, adv_shape)
        entropies = dist.entropy_heads()
        if len(entropies) != len(self.head_entropy_coeffs):
            raise ValueError("One entropy coefficient is required per action head")
        self._head_entropies = [entropy.detach().mean() for entropy in entropies]
        weighted = sum(
            coeff * entropy
            for coeff, entropy in zip(self.head_entropy_coeffs, entropies)
        )
        return weighted.unsqueeze(-1)

    def forward(self, tensordict):
        self._head_entropies = None
        out = super().forward(tensordict)
        for head, entropy in enumerate(self._head_entropies or ()):
            out.set(("entropy_movement", "entropy_message")[head], entropy)
        return out


def _factorized_policy(algorithm: Algorithm, group: str, model_config: ModelConfig):
    """Two head actor over a MultiDiscrete [motor, message] action"""
    action_spec = algorithm.action_spec[group, "action"]
    nvec = getattr(action_spec, "nvec", None)
    if nvec is None or len(action_spec.shape) < 1 or action_spec.shape[-1] != 2:
        raise ValueError("Actions must be [motor, message] pairs")
    nvec = [int(size) for size in nvec.reshape(-1, 2)[0].tolist()]
    n_agents = len(algorithm.group_map[group])
    observation_spec = algorithm.observation_spec[group]
    if "features" not in observation_spec.keys():
        raise ValueError("The factorized policy needs the features observation")

    actor_module = model_config.get_model(
        input_spec=Composite({group: observation_spec.clone().to(algorithm.device)}),
        output_spec=Composite(
            {
                group: Composite(
                    {"logits": Unbounded(shape=(n_agents, sum(nvec)))},
                    shape=(n_agents,),
                )
            }
        ),
        agent_group=group,
        input_has_agent_dim=True,
        n_agents=n_agents,
        centralised=False,
        share_params=algorithm.experiment_config.share_policy_params,
        device=algorithm.device,
        action_spec=algorithm.action_spec,
    )
    informer_can_move = bool(getattr(algorithm, "informer_can_move", False))
    mask_module = TensorDictModule(
        lambda logits, features: mask_factorized_logits(
            logits, features, *nvec, informer_can_move=informer_can_move
        ),
        in_keys=[(group, "logits"), (group, "features")],
        out_keys=[(group, "logits")],
    )
    return ProbabilisticActor(
        module=TensorDictSequential(actor_module, mask_module),
        spec=action_spec,
        in_keys=[(group, "logits")],
        out_keys=[(group, "action")],
        distribution_class=FactorizedCategorical,
        distribution_kwargs={"nvec": nvec},
        return_log_prob=True,
        log_prob_key=(group, "log_prob"),
    )


def _factorized_loss(algorithm: Algorithm, group: str, policy_for_loss):
    loss_module = FactorizedClipPPOLoss(
        actor=policy_for_loss,
        critic=algorithm.get_critic(group),
        clip_epsilon=algorithm.clip_epsilon,
        head_entropy_coeffs=(algorithm.entropy_coef, algorithm.message_entropy_coef),
        critic_coeff=algorithm.critic_coef,
        loss_critic_type=algorithm.loss_critic_type,
        normalize_advantage=False,
    )
    loss_module.set_keys(
        reward=(group, "reward"),
        action=(group, "action"),
        done=(group, "done"),
        terminated=(group, "terminated"),
        advantage=(group, "advantage"),
        value_target=(group, "value_target"),
        value=(group, "state_value"),
        sample_log_prob=(group, "log_prob"),
    )
    loss_module.make_value_estimator(
        ValueEstimators.GAE,
        gamma=algorithm.experiment_config.gamma,
        lmbda=algorithm.lmbda,
    )
    return loss_module, False


class FactorizedMappo(Mappo):
    """MAPPO with factorized heads"""

    def __init__(
        self, message_entropy_coef: float, informer_can_move: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.message_entropy_coef = message_entropy_coef
        self.informer_can_move = informer_can_move

    def _get_policy_for_loss(self, group, model_config, continuous):
        if continuous:
            raise ValueError("The factorized policy is discrete")
        return _factorized_policy(self, group, model_config)

    def _get_loss(self, group, policy_for_loss, continuous):
        return _factorized_loss(self, group, policy_for_loss)


class FactorizedIppo(Ippo):
    """IPPO with factorized heads"""

    def __init__(
        self, message_entropy_coef: float, informer_can_move: bool = False, **kwargs
    ):
        super().__init__(**kwargs)
        self.message_entropy_coef = message_entropy_coef
        self.informer_can_move = informer_can_move

    def _get_policy_for_loss(self, group, model_config, continuous):
        if continuous:
            raise ValueError("The factorized policy is discrete")
        return _factorized_policy(self, group, model_config)

    def _get_loss(self, group, policy_for_loss, continuous):
        return _factorized_loss(self, group, policy_for_loss)


@dataclass
class FactorizedMappoConfig(MappoConfig):
    message_entropy_coef: float = MISSING
    informer_can_move: bool = False

    @classmethod
    def get_from_yaml(cls, path=None):
        # The shipped yaml is keyed by the wrapped algorithm's name.
        base = MappoConfig.get_from_yaml(path)
        return cls(**base.__dict__, message_entropy_coef=base.entropy_coef)

    @staticmethod
    def associated_class() -> type[Algorithm]:
        return FactorizedMappo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False


@dataclass
class FactorizedIppoConfig(IppoConfig):
    message_entropy_coef: float = MISSING
    informer_can_move: bool = False

    @classmethod
    def get_from_yaml(cls, path=None):
        base = IppoConfig.get_from_yaml(path)
        return cls(**base.__dict__, message_entropy_coef=base.entropy_coef)

    @staticmethod
    def associated_class() -> type[Algorithm]:
        return FactorizedIppo

    @staticmethod
    def supports_continuous_actions() -> bool:
        return False
