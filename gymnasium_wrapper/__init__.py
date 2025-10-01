from gymnasium.envs.registration import register


register(
    id="VizdoomBasic-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "basic.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomBasic-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "basic.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomBasicAudio-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "basic_audio.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomBasicAudio-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "basic_audio.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomCorridor-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "deadly_corridor.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomCorridor-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "deadly_corridor.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomDefendCenter-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "defend_the_center.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomDefendCenter-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "defend_the_center.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomDefendLine-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "defend_the_line.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomDefendLine-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "defend_the_line.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomHealthGathering-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "health_gathering.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomHealthGathering-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "health_gathering.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomMyWayHome-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "my_way_home.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomMyWayHome-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "my_way_home.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomPredictPosition-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "predict_position.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomPredictPosition-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "predict_position.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomTakeCover-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "take_cover.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomTakeCover-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "take_cover.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomDeathmatch-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "deathmatch.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomDeathmatch-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={"scenario_config_file": "deathmatch.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomHealthGatheringSupreme-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={
        "scenario_config_file": "health_gathering_supreme.cfg",
        "max_buttons_pressed": 1,
    },
)

register(
    id="VizdoomHealthGatheringSupreme-MultiBinary-v1",
    entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv",
    kwargs={
        "scenario_config_file": "health_gathering_supreme.cfg",
        "max_buttons_pressed": 0,
    },
)

# register(
#     id="VizdoomFreedoom1-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomFullGameEnv",
#     kwargs={"scenario_config_file": "freedoom1.cfg", "max_buttons_pressed": 0},
# )

# register(
#     id="VizdoomFreedoom2-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomFullGameEnv",
#     kwargs={"scenario_config_file": "freedoom2.cfg", "max_buttons_pressed": 0},
# )

# register(
#     id="VizdoomDoom-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomFullGameEnv",
#     kwargs={"scenario_config_file": "doom.cfg", "max_buttons_pressed": 0},
# )

# register(
#     id="VizdoomDoom2-v0",
#     entry_point="vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomFullGameEnv",
#     kwargs={"scenario_config_file": "doom2.cfg", "max_buttons_pressed": 0},
# )
