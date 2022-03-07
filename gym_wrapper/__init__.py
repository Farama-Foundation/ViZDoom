from gym.envs.registration import register

register(id="VizdoomBasic-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomBasic")

register(id="VizdoomCorridor-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomCorridor")

register(
    id="VizdoomDefendCenter-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomDefendCenter"
)

register(id="VizdoomDefendLine-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomDefendLine")

register(
    id="VizdoomHealthGathering-v0",
    entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomHealthGathering",
)

register(id="VizdoomMyWayHome-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomMyWayHome")

register(
    id="VizdoomPredictPosition-v0",
    entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomPredictPosition",
)

register(id="VizdoomTakeCover-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomTakeCover")

register(id="VizdoomDeathmatch-v0", entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomDeathmatch")

register(
    id="VizdoomHealthGatheringSupreme-v0",
    entry_point="vizdoom.gym_wrapper.gym_env_defns:VizdoomHealthGatheringSupreme",
)
