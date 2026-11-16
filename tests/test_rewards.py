# Tests reward settings, game events, accumulation, and resets across episodes.

import os

import pytest

import vizdoom as vzd


@pytest.mark.parametrize("multiplayer", [False, True])
@pytest.mark.parametrize("penalty", [10, -10])
def test_death_penalty_once_per_death(multiplayer, penalty):
    game = vzd.DoomGame()
    scenario = "multi.cfg" if multiplayer else "basic.cfg"
    game.load_config(os.path.join(vzd.scenarios_path, scenario))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_available_buttons([])
    game.set_living_reward(0)
    game.set_death_penalty(penalty)
    if multiplayer:
        game.add_game_args("-host 1 -deathmatch +sv_cheats 1")
    game.init()

    try:
        for _ in range(2):
            assert game.get_last_reward() == 0
            assert game.get_total_reward() == 0
            game.send_game_command("kill")
            assert game.make_action([]) == -penalty
            assert game.is_player_dead()
            assert game.get_total_reward() == -penalty

            if multiplayer:
                for tics in (1, 4, 8):
                    assert game.make_action([], tics) == 0
                    assert game.is_player_dead()
                    assert game.get_total_reward() == -penalty

                game.respawn_player()
                assert game.get_last_reward() == 0
                assert game.get_total_reward() == -penalty
                game.send_game_command("kill")
                assert game.make_action([]) == -penalty
                assert game.get_total_reward() == -2 * penalty

            game.new_episode()
    finally:
        game.close()


@pytest.mark.parametrize("deaths", [1, 2])
def test_death_penalty_with_automatic_respawn(deaths):
    game = vzd.DoomGame()
    game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_available_buttons([])
    game.set_living_reward(0)
    game.set_death_penalty(10)
    game.add_game_args(
        "-host 1 -deathmatch +sv_cheats 1 +sv_forcerespawn 1 +viz_respawn_delay 0"
    )
    game.init()

    try:
        for _ in range(deaths):
            game.send_game_command("kill")
            game.advance_action(70, update_state=False)
            assert not game.is_player_dead()
        assert game.get_game_variable(vzd.GameVariable.DEATHCOUNT) == deaths

        assert game.make_action([]) == -10 * deaths
        assert game.get_total_reward() == -10 * deaths
        assert game.make_action([], 4) == 0
        assert game.get_total_reward() == -10 * deaths
    finally:
        game.close()


REWARD_SETTINGS = (
    "living_reward",
    "death_penalty",
    "death_reward",
    "map_exit_reward",
    "kill_reward",
    "secret_reward",
    "item_reward",
    "frag_reward",
    "hit_reward",
    "hit_taken_penalty",
    "hit_taken_reward",
    "damage_made_reward",
    "damage_taken_penalty",
    "damage_taken_reward",
    "health_reward",
    "armor_reward",
)


@pytest.fixture
def reward_game():
    game = vzd.DoomGame()
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
    game.set_doom_skill(3)
    game.set_seed(7)
    game.set_available_buttons([])
    # Use the bundled game's first map without monsters or scenario ACS rewards.
    game.add_game_args("-nomonsters")
    try:
        yield game
    finally:
        game.close()


@pytest.mark.parametrize("setting", REWARD_SETTINGS)
def test_reward_settings_before_and_after_init(reward_game, setting):
    game = reward_game
    get_reward = getattr(game, f"get_{setting}")
    set_reward = getattr(game, f"set_{setting}")
    assert get_reward() == 0
    set_reward(1.25)
    game.init()
    assert get_reward() == 1.25

    for value in (-2.5, 0, 3.75):
        set_reward(value)
        assert get_reward() == value
        game.new_episode()
        assert get_reward() == value
        assert game.get_last_reward() == 0
        assert game.get_total_reward() == 0


@pytest.mark.parametrize("name", ["death", "hit_taken", "damage_taken"])
@pytest.mark.parametrize("value", [-2.5, 0, 1.25])
def test_reward_and_penalty_are_opposite_aliases(reward_game, name, value):
    game = reward_game
    getattr(game, f"set_{name}_penalty")(value)
    assert getattr(game, f"get_{name}_reward")() == -value
    getattr(game, f"set_{name}_reward")(value)
    assert getattr(game, f"get_{name}_penalty")() == -value


@pytest.mark.parametrize("living_reward", [-0.25, 0, 1.5])
@pytest.mark.parametrize("start_time", [1, 14])
def test_living_reward_per_tic_and_episode_reset(
    reward_game, living_reward, start_time
):
    game = reward_game
    game.set_episode_start_time(start_time)
    game.set_living_reward(living_reward)
    game.init()

    for _ in range(2):
        assert game.get_last_reward() == 0
        assert game.get_total_reward() == 0
        total = 0
        for tics in (1, 4, 8):
            assert game.make_action([], tics) == pytest.approx(tics * living_reward)
            total += tics * living_reward
            assert game.get_last_reward() == pytest.approx(tics * living_reward)
            assert game.get_total_reward() == pytest.approx(total)
        game.new_episode()


def test_living_reward_changes_during_episode(reward_game):
    game = reward_game
    game.set_living_reward(0.5)
    game.init()
    assert game.make_action([], 4) == 2
    game.set_living_reward(-1.25)
    assert game.make_action([], 2) == -2.5
    assert game.get_total_reward() == -0.5
    game.set_living_reward(0)
    assert game.make_action([], 4) == 0
    assert game.get_total_reward() == -0.5


def test_rewards_wait_for_state_update(reward_game):
    game = reward_game
    game.set_living_reward(0.5)
    game.set_health_reward(2)
    game.init()
    assert game.make_action([], 2) == 1
    game.send_game_command("give HealthBonus")
    game.advance_action(3, update_state=False)
    game.advance_action(4, update_state=False)
    assert game.get_last_reward() == 1
    assert game.get_total_reward() == 1

    game.advance_action(1)
    assert game.get_last_reward() == 8 * 0.5 + 2
    assert game.get_total_reward() == 1 + 8 * 0.5 + 2
    assert game.make_action([]) == 0.5
    assert game.get_total_reward() == 7.5


@pytest.mark.parametrize("living_reward", [-0.25, 1.5])
def test_living_reward_stops_at_episode_timeout(reward_game, living_reward):
    game = reward_game
    game.set_episode_start_time(1)
    game.set_episode_timeout(10)
    game.set_living_reward(living_reward)
    game.set_map_exit_reward(100)
    game.init()
    start = game.get_episode_time()
    reward = game.make_action([], 100)
    elapsed = game.get_episode_time() - start
    assert 0 < elapsed < 100
    assert game.is_episode_finished()
    assert reward == pytest.approx(elapsed * living_reward)
    assert game.get_last_reward() == pytest.approx(reward)
    assert game.get_total_reward() == pytest.approx(reward)


# The special commands produce real engine events with known counter changes.
# Give does not count as an item pickup; that is tested separately below.
EVENT_REWARDS = [
    (
        "kill_reward",
        vzd.GameVariable.KILLCOUNT,
        "summon DoomImp 0 42",
        "special Thing_Damage 42 1000",
        1,
    ),
    (
        "secret_reward",
        vzd.GameVariable.SECRETCOUNT,
        "summon SecretTrigger 0 42",
        "special Thing_Activate 42",
        1,
    ),
    (
        "hit_reward",
        vzd.GameVariable.HITCOUNT,
        "summon DoomImp 0 42",
        "special Thing_Damage 42 7",
        1,
    ),
    (
        "damage_made_reward",
        vzd.GameVariable.DAMAGECOUNT,
        "summon DoomImp 0 42",
        "special Thing_Damage 42 7",
        7,
    ),
    (
        "hit_taken_reward",
        vzd.GameVariable.HITS_TAKEN,
        None,
        "special Thing_Damage 0 7",
        1,
    ),
    (
        "hit_taken_penalty",
        vzd.GameVariable.HITS_TAKEN,
        None,
        "special Thing_Damage 0 7",
        1,
    ),
    (
        "damage_taken_reward",
        vzd.GameVariable.DAMAGE_TAKEN,
        None,
        "special Thing_Damage 0 7",
        7,
    ),
    (
        "damage_taken_penalty",
        vzd.GameVariable.DAMAGE_TAKEN,
        None,
        "special Thing_Damage 0 7",
        7,
    ),
    ("health_reward", vzd.GameVariable.HEALTH, None, "give HealthBonus", 1),
    ("armor_reward", vzd.GameVariable.ARMOR, None, "give ArmorBonus", 1),
]


@pytest.mark.parametrize("value", [-2.5, 0, 1.25])
@pytest.mark.parametrize(
    "setting,variable,setup,command,delta",
    EVENT_REWARDS,
    ids=[event[0] for event in EVENT_REWARDS],
)
def test_event_rewards(reward_game, setting, variable, setup, command, delta, value):
    game = reward_game
    getattr(game, f"set_{setting}")(value)
    game.init()
    expected = delta * value * (-1 if setting.endswith("penalty") else 1)

    for _ in range(2):
        assert game.get_last_reward() == 0
        assert game.get_total_reward() == 0
        if setup:
            game.send_game_command(setup)
            assert game.make_action([]) == 0
        before = game.get_game_variable(variable)
        game.send_game_command(command)
        assert game.make_action([]) == pytest.approx(expected)
        assert game.get_game_variable(variable) - before == delta
        assert game.get_last_reward() == pytest.approx(expected)
        assert game.get_total_reward() == pytest.approx(expected)
        assert game.make_action([]) == 0
        assert game.get_total_reward() == pytest.approx(expected)
        game.new_episode()


@pytest.mark.parametrize("value", [-2.5, 0, 1.25])
def test_item_reward_for_actual_pickup(reward_game, value):
    game = reward_game
    game.set_available_buttons([vzd.Button.MOVE_FORWARD])
    game.set_item_reward(value)
    game.init()
    for _ in range(2):
        game.send_game_command("summon HealthBonus")
        for _ in range(20):
            reward = game.make_action([1])
            if game.get_game_variable(vzd.GameVariable.ITEMCOUNT) == 1:
                assert reward == pytest.approx(value)
                break
            assert reward == 0
        else:
            pytest.fail("HealthBonus was not picked up")
        assert game.get_last_reward() == pytest.approx(value)
        assert game.get_total_reward() == pytest.approx(value)
        assert game.make_action([0]) == 0
        assert game.get_total_reward() == pytest.approx(value)
        game.new_episode()


@pytest.mark.parametrize(
    "setting,variable,give,take",
    [
        (
            "health_reward",
            vzd.GameVariable.HEALTH,
            "give HealthBonus",
            "special Thing_Damage 0 7",
        ),
        ("armor_reward", vzd.GameVariable.ARMOR, "give ArmorBonus", "take armor"),
    ],
)
def test_health_and_armor_reward_only_increases(
    reward_game, setting, variable, give, take
):
    game = reward_game
    getattr(game, f"set_{setting}")(2)
    game.init()
    game.send_game_command(give)
    assert game.make_action([]) == 2
    before = game.get_game_variable(variable)
    game.send_game_command(take)
    assert game.make_action([]) == 0
    assert game.get_game_variable(variable) < before
    game.send_game_command(give)
    assert game.make_action([]) == 2
    assert game.get_total_reward() == 4


def test_enabling_reward_does_not_reward_past_events(reward_game):
    game = reward_game
    game.init()
    game.send_game_command("special Thing_Damage 0 7")
    assert game.make_action([]) == 0
    assert game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN) == 7
    game.set_damage_taken_reward(2)
    assert game.make_action([]) == 0
    game.send_game_command("special Thing_Damage 0 3")
    assert game.make_action([]) == 6
    assert game.get_total_reward() == 6


def test_combined_rewards_and_multiple_events(reward_game):
    game = reward_game
    game.set_living_reward(0.5)
    game.set_health_reward(2)
    game.set_armor_reward(3)
    game.set_hit_taken_penalty(5)
    game.set_damage_taken_penalty(0.25)
    game.init()
    game.send_game_command("give HealthBonus; give HealthBonus; give ArmorBonus")
    assert game.make_action([], 4) == 4 * 0.5 + 2 * 2 + 3
    assert game.get_total_reward() == 9

    game.send_game_command("special Thing_Damage 0 7; special Thing_Damage 0 3")
    reward = game.make_action([])
    damage = game.get_game_variable(vzd.GameVariable.DAMAGE_TAKEN)
    assert damage > 0
    assert game.get_game_variable(vzd.GameVariable.HITS_TAKEN) == 2
    expected = 0.5 - 2 * 5 - damage * 0.25
    assert reward == pytest.approx(expected)
    assert game.get_last_reward() == pytest.approx(expected)
    assert game.get_total_reward() == pytest.approx(9 + expected)


@pytest.mark.parametrize("exit_reward", [-2.5, 0, 10])
def test_map_script_rewards_and_map_exit(reward_game, exit_reward):
    game = reward_game
    game.load_config(os.path.join(vzd.scenarios_path, "basic.cfg"))
    game.set_living_reward(0)
    game.set_map_exit_reward(exit_reward)
    game.set_doom_skill(3)
    game.set_available_buttons([])
    game.init()
    for _ in range(2):
        assert game.get_last_reward() == 0
        assert game.get_total_reward() == 0
        # basic.wad subtracts 5 from its ACS reward when ammo decreases.
        game.send_game_command("take Clip 1")
        assert game.make_action([], 2) == -5
        assert game.get_total_reward() == -5
        assert game.make_action([]) == 0
        # Script 4 adds 106 to the map reward and calls Exit_Normal.
        game.send_game_command("puke 4")
        assert game.make_action([], 2) == 106 + exit_reward
        assert game.is_episode_finished()
        assert not game.is_player_dead()
        assert game.get_last_reward() == 106 + exit_reward
        assert game.get_total_reward() == 101 + exit_reward
        game.new_episode()


def test_death_reward_does_not_grant_map_exit_reward(reward_game):
    game = reward_game
    game.set_death_reward(-3.5)
    game.set_map_exit_reward(100)
    game.init()
    game.send_game_command("kill")
    assert game.make_action([]) == -3.5
    assert game.is_player_dead()
    assert game.is_episode_finished()
    assert game.get_last_reward() == -3.5
    assert game.get_total_reward() == -3.5


@pytest.mark.parametrize("value", [-2.5, 0, 1.25])
def test_frag_reward(reward_game, tmp_path, monkeypatch, value):
    game = reward_game
    monkeypatch.chdir(tmp_path)
    bots_path = tmp_path / "bots.cfg"
    bots_path.write_text("{ name RewardTarget aiming 0 perfection 0 reaction 0 isp 0 }")
    game.load_config(os.path.join(vzd.scenarios_path, "multi.cfg"))
    game.set_mode(vzd.Mode.PLAYER)
    game.set_window_visible(False)
    game.set_available_buttons([])
    game.set_death_penalty(0)
    game.set_frag_reward(value)
    game.add_game_args("-host 1 -deathmatch +sv_cheats 1 +viz_bots_path bots.cfg")
    game.init()
    game.send_game_command("god; addbot")
    assert game.make_action([], 5) == 0
    assert game.get_game_variable(vzd.GameVariable.PLAYER_COUNT) == 2
    # multi_deathmatch.wad assigns TID 1001 to the second player on entry.
    game.send_game_command("special Thing_Damage 1001 1000")
    assert game.make_action([]) == pytest.approx(value)
    assert game.get_game_variable(vzd.GameVariable.FRAGCOUNT) == 1
    assert game.get_last_reward() == pytest.approx(value)
    assert game.get_total_reward() == pytest.approx(value)
    assert game.make_action([]) == 0
    assert game.get_total_reward() == pytest.approx(value)
