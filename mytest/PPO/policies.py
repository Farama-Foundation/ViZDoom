# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from .common.policies import (
    ActorCriticCnnPolicy,
    ActorCriticPolicy,
    MultiInputActorCriticPolicy,
    ActorCriticCnnLSTMPolicy,
    register_policy,
)

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
CnnLSTMPolicy = ActorCriticCnnLSTMPolicy

register_policy("MlpPolicy", ActorCriticPolicy)
register_policy("CnnPolicy", ActorCriticCnnPolicy)
register_policy("MultiInputPolicy", MultiInputPolicy)
register_policy("CnnLSTMPolicy", ActorCriticCnnLSTMPolicy)
