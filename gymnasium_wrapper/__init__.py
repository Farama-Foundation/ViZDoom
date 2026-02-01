from gymnasium.envs.registration import register


DEFAULT_VIZDOOM_ENTRYPOINT = (
    "vizdoom.gymnasium_wrapper.gymnasium_env_defns:VizdoomScenarioEnv"
)

register(
    id="VizdoomBasic-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "basic.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomBasic-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "basic.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomBasicAudio-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "basic_audio.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomBasicAudio-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "basic_audio.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomBasicNotifications-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={
        "scenario_config_file": "basic_notifications.cfg",
        "max_buttons_pressed": 1,
    },
)

register(
    id="VizdoomBasicNotifications-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={
        "scenario_config_file": "basic_notifications.cfg",
        "max_buttons_pressed": 0,
    },
)

register(
    id="VizdoomCorridor-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "deadly_corridor.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomCorridor-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "deadly_corridor.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomDefendCenter-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "defend_the_center.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomDefendCenter-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "defend_the_center.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomDefendLine-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "defend_the_line.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomDefendLine-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "defend_the_line.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomHealthGathering-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "health_gathering.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomHealthGathering-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "health_gathering.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomMyWayHome-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "my_way_home.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomMyWayHome-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "my_way_home.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomPredictPosition-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "predict_position.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomPredictPosition-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "predict_position.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomTakeCover-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "take_cover.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomTakeCover-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "take_cover.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomDeathmatch-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "deathmatch.cfg", "max_buttons_pressed": 1},
)

register(
    id="VizdoomDeathmatch-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={"scenario_config_file": "deathmatch.cfg", "max_buttons_pressed": 0},
)

register(
    id="VizdoomHealthGatheringSupreme-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={
        "scenario_config_file": "health_gathering_supreme.cfg",
        "max_buttons_pressed": 1,
    },
)

register(
    id="VizdoomHealthGatheringSupreme-MultiBinary-v1",
    entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
    kwargs={
        "scenario_config_file": "health_gathering_supreme.cfg",
        "max_buttons_pressed": 0,
    },
)

SKILL_LEVELS = ["S1", "S2", "S3", "S4", "S5"]

DOOM_MAPS = [
    "E1M1",
    "E1M2",
    "E1M3",
    "E1M4",
    "E1M5",
    "E1M6",
    "E1M7",
    "E1M8",
    "E1M9",
    "E2M1",
    "E2M2",
    "E2M3",
    "E2M4",
    "E2M5",
    "E2M6",
    "E2M7",
    "E2M8",
    "E2M9",
    "E3M1",
    "E3M2",
    "E3M3",
    "E3M4",
    "E3M5",
    "E3M6",
    "E3M7",
    "E3M8",
    "E3M9",
    "E4M1",
    "E4M2",
    "E4M3",
    "E4M4",
    "E4M5",
    "E4M6",
    "E4M7",
    "E4M8",
    "E4M9",
]

DOOM2_MAPS = [
    "MAP01",
    "MAP02",
    "MAP03",
    "MAP04",
    "MAP05",
    "MAP06",
    "MAP07",
    "MAP08",
    "MAP09",
    "MAP10",
    "MAP11",
    "MAP12",
    "MAP13",
    "MAP14",
    "MAP15",
    "MAP16",
    "MAP17",
    "MAP18",
    "MAP19",
    "MAP20",
    "MAP21",
    "MAP22",
    "MAP23",
    "MAP24",
    "MAP25",
    "MAP26",
    "MAP27",
    "MAP28",
    "MAP29",
    "MAP30",
    "MAP31",
    "MAP32",
]

for skill in SKILL_LEVELS:
    skill_int = int(skill[1])
    for game_name, config_file, maps in [
        ("Doom", "doom.cfg", DOOM_MAPS),
        ("Doom2", "doom2.cfg", DOOM2_MAPS),
        ("Freedoom1", "freedoom1.cfg", DOOM_MAPS),
        ("Freedoom2", "freedoom2.cfg", DOOM2_MAPS),
    ]:
        for map in maps:
            register(
                id=f"Vizdoom{game_name}{map}-{skill}-v0",
                entry_point=DEFAULT_VIZDOOM_ENTRYPOINT,
                kwargs={
                    "scenario_config_file": f"{config_file}",
                    "max_buttons_pressed": 0,
                    "doom_map": map,
                    "doom_skill": skill_int,
                },
            )
