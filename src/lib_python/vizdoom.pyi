"""
ViZDoom Python Type Stubs

This file provides type information for static analysis and IDE support.
Auto-generated via pybind11-stubgen and formatted with black, isort.

For the official documentation, see: https://vizdoom.farama.org/
"""


from __future__ import annotations

import typing

from numpy.typing import NDArray

__all__: list[str] = [
    "ABGR32",
    "ACTIVATE_SELECTED_ITEM",
    "ALTATTACK",
    "ALTATTACK_READY",
    "AMMO0",
    "AMMO1",
    "AMMO2",
    "AMMO3",
    "AMMO4",
    "AMMO5",
    "AMMO6",
    "AMMO7",
    "AMMO8",
    "AMMO9",
    "ANGLE",
    "ARGB32",
    "ARMOR",
    "ASYNC_PLAYER",
    "ASYNC_SPECTATOR",
    "ATTACK",
    "ATTACK_READY",
    "AutomapMode",
    "BGR24",
    "BGRA32",
    "BINARY_BUTTON_COUNT",
    "BUTTON_COUNT",
    "Button",
    "CAMERA_ANGLE",
    "CAMERA_FOV",
    "CAMERA_PITCH",
    "CAMERA_POSITION_X",
    "CAMERA_POSITION_Y",
    "CAMERA_POSITION_Z",
    "CAMERA_ROLL",
    "CBCGCR",
    "CRCGCB",
    "CROUCH",
    "DAMAGECOUNT",
    "DAMAGE_TAKEN",
    "DEAD",
    "DEATHCOUNT",
    "DEFAULT_FPS",
    "DEFAULT_FRAMETIME_MS",
    "DEFAULT_FRAMETIME_S",
    "DEFAULT_TICRATE",
    "DELTA_BUTTON_COUNT",
    "DOOM_256_COLORS8",
    "DROP_SELECTED_ITEM",
    "DROP_SELECTED_WEAPON",
    "DoomGame",
    "FRAGCOUNT",
    "FileDoesNotExistException",
    "GRAY8",
    "GameState",
    "GameVariable",
    "HEALTH",
    "HITCOUNT",
    "HITS_TAKEN",
    "ITEMCOUNT",
    "JUMP",
    "KILLCOUNT",
    "LAND",
    "LOOK_DOWN",
    "LOOK_UP",
    "LOOK_UP_DOWN_DELTA",
    "Label",
    "Line",
    "MAX_PLAYERS",
    "MAX_PLAYER_NAME_LENGTH",
    "MOVE_BACKWARD",
    "MOVE_DOWN",
    "MOVE_FORWARD",
    "MOVE_FORWARD_BACKWARD_DELTA",
    "MOVE_LEFT",
    "MOVE_LEFT_RIGHT_DELTA",
    "MOVE_RIGHT",
    "MOVE_UP",
    "MOVE_UP_DOWN_DELTA",
    "MessageQueueException",
    "Mode",
    "NORMAL",
    "OBJECTS",
    "OBJECTS_WITH_SIZE",
    "ON_GROUND",
    "Object",
    "PITCH",
    "PLAYER",
    "PLAYER10_FRAGCOUNT",
    "PLAYER11_FRAGCOUNT",
    "PLAYER12_FRAGCOUNT",
    "PLAYER13_FRAGCOUNT",
    "PLAYER14_FRAGCOUNT",
    "PLAYER15_FRAGCOUNT",
    "PLAYER16_FRAGCOUNT",
    "PLAYER1_FRAGCOUNT",
    "PLAYER2_FRAGCOUNT",
    "PLAYER3_FRAGCOUNT",
    "PLAYER4_FRAGCOUNT",
    "PLAYER5_FRAGCOUNT",
    "PLAYER6_FRAGCOUNT",
    "PLAYER7_FRAGCOUNT",
    "PLAYER8_FRAGCOUNT",
    "PLAYER9_FRAGCOUNT",
    "PLAYER_COUNT",
    "PLAYER_NUMBER",
    "POSITION_X",
    "POSITION_Y",
    "POSITION_Z",
    "RELOAD",
    "RES_1024X576",
    "RES_1024X640",
    "RES_1024X768",
    "RES_1280X1024",
    "RES_1280X720",
    "RES_1280X800",
    "RES_1280X960",
    "RES_1400X1050",
    "RES_1400X787",
    "RES_1400X875",
    "RES_1600X1000",
    "RES_1600X1200",
    "RES_1600X900",
    "RES_160X120",
    "RES_1920X1080",
    "RES_200X125",
    "RES_200X150",
    "RES_256X144",
    "RES_256X160",
    "RES_256X192",
    "RES_320X180",
    "RES_320X200",
    "RES_320X240",
    "RES_320X256",
    "RES_400X225",
    "RES_400X250",
    "RES_400X300",
    "RES_512X288",
    "RES_512X320",
    "RES_512X384",
    "RES_640X360",
    "RES_640X400",
    "RES_640X480",
    "RES_800X450",
    "RES_800X500",
    "RES_800X600",
    "RGB24",
    "RGBA32",
    "ROLL",
    "SECRETCOUNT",
    "SELECTED_WEAPON",
    "SELECTED_WEAPON_AMMO",
    "SELECT_NEXT_ITEM",
    "SELECT_NEXT_WEAPON",
    "SELECT_PREV_ITEM",
    "SELECT_PREV_WEAPON",
    "SELECT_WEAPON0",
    "SELECT_WEAPON1",
    "SELECT_WEAPON2",
    "SELECT_WEAPON3",
    "SELECT_WEAPON4",
    "SELECT_WEAPON5",
    "SELECT_WEAPON6",
    "SELECT_WEAPON7",
    "SELECT_WEAPON8",
    "SELECT_WEAPON9",
    "SLOT_COUNT",
    "SPECTATOR",
    "SPEED",
    "SR_11025",
    "SR_22050",
    "SR_44100",
    "STRAFE",
    "SamplingRate",
    "ScreenFormat",
    "ScreenResolution",
    "Sector",
    "ServerState",
    "SharedMemoryException",
    "TURN180",
    "TURN_LEFT",
    "TURN_LEFT_RIGHT_DELTA",
    "TURN_RIGHT",
    "USE",
    "USER1",
    "USER10",
    "USER11",
    "USER12",
    "USER13",
    "USER14",
    "USER15",
    "USER16",
    "USER17",
    "USER18",
    "USER19",
    "USER2",
    "USER20",
    "USER21",
    "USER22",
    "USER23",
    "USER24",
    "USER25",
    "USER26",
    "USER27",
    "USER28",
    "USER29",
    "USER3",
    "USER30",
    "USER31",
    "USER32",
    "USER33",
    "USER34",
    "USER35",
    "USER36",
    "USER37",
    "USER38",
    "USER39",
    "USER4",
    "USER40",
    "USER41",
    "USER42",
    "USER43",
    "USER44",
    "USER45",
    "USER46",
    "USER47",
    "USER48",
    "USER49",
    "USER5",
    "USER50",
    "USER51",
    "USER52",
    "USER53",
    "USER54",
    "USER55",
    "USER56",
    "USER57",
    "USER58",
    "USER59",
    "USER6",
    "USER60",
    "USER7",
    "USER8",
    "USER9",
    "USER_VARIABLE_COUNT",
    "VELOCITY_X",
    "VELOCITY_Y",
    "VELOCITY_Z",
    "VIEW_HEIGHT",
    "ViZDoomErrorException",
    "ViZDoomIsNotRunningException",
    "ViZDoomNoOpenALSoundException",
    "ViZDoomUnexpectedExitException",
    "WEAPON0",
    "WEAPON1",
    "WEAPON2",
    "WEAPON3",
    "WEAPON4",
    "WEAPON5",
    "WEAPON6",
    "WEAPON7",
    "WEAPON8",
    "WEAPON9",
    "WHOLE",
    "ZOOM",
    "doom_fixed_to_double",
    "doom_fixed_to_float",
    "doom_tics_to_ms",
    "doom_tics_to_sec",
    "get_default_categories",
    "is_binary_button",
    "is_delta_button",
    "ms_to_doom_tics",
    "sec_to_doom_tics",
]

class AutomapMode:
    """
    Defines the automap rendering mode.

    Members:

      NORMAL

      WHOLE

      OBJECTS

      OBJECTS_WITH_SIZE
    """

    NORMAL: typing.ClassVar[AutomapMode]  # value = <AutomapMode.NORMAL: 0>
    OBJECTS: typing.ClassVar[AutomapMode]  # value = <AutomapMode.OBJECTS: 2>
    OBJECTS_WITH_SIZE: typing.ClassVar[
        AutomapMode
    ]  # value = <AutomapMode.OBJECTS_WITH_SIZE: 3>
    WHOLE: typing.ClassVar[AutomapMode]  # value = <AutomapMode.WHOLE: 1>
    __members__: typing.ClassVar[
        dict[str, AutomapMode]
    ]  # value = {'NORMAL': <AutomapMode.NORMAL: 0>, 'WHOLE': <AutomapMode.WHOLE: 1>, 'OBJECTS': <AutomapMode.OBJECTS: 2>, 'OBJECTS_WITH_SIZE': <AutomapMode.OBJECTS_WITH_SIZE: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Button:
    """
    Defines available game buttons/actions that can be used to control the game.

    Members:

      ATTACK

      USE

      JUMP

      CROUCH

      TURN180

      ALTATTACK

      RELOAD

      ZOOM

      SPEED

      STRAFE

      MOVE_RIGHT

      MOVE_LEFT

      MOVE_BACKWARD

      MOVE_FORWARD

      TURN_RIGHT

      TURN_LEFT

      LOOK_UP

      LOOK_DOWN

      MOVE_UP

      MOVE_DOWN

      LAND

      SELECT_WEAPON1

      SELECT_WEAPON2

      SELECT_WEAPON3

      SELECT_WEAPON4

      SELECT_WEAPON5

      SELECT_WEAPON6

      SELECT_WEAPON7

      SELECT_WEAPON8

      SELECT_WEAPON9

      SELECT_WEAPON0

      SELECT_NEXT_WEAPON

      SELECT_PREV_WEAPON

      DROP_SELECTED_WEAPON

      ACTIVATE_SELECTED_ITEM

      SELECT_NEXT_ITEM

      SELECT_PREV_ITEM

      DROP_SELECTED_ITEM

      LOOK_UP_DOWN_DELTA

      TURN_LEFT_RIGHT_DELTA

      MOVE_FORWARD_BACKWARD_DELTA

      MOVE_LEFT_RIGHT_DELTA

      MOVE_UP_DOWN_DELTA
    """

    ACTIVATE_SELECTED_ITEM: typing.ClassVar[
        Button
    ]  # value = <Button.ACTIVATE_SELECTED_ITEM: 34>
    ALTATTACK: typing.ClassVar[Button]  # value = <Button.ALTATTACK: 5>
    ATTACK: typing.ClassVar[Button]  # value = <Button.ATTACK: 0>
    CROUCH: typing.ClassVar[Button]  # value = <Button.CROUCH: 3>
    DROP_SELECTED_ITEM: typing.ClassVar[
        Button
    ]  # value = <Button.DROP_SELECTED_ITEM: 37>
    DROP_SELECTED_WEAPON: typing.ClassVar[
        Button
    ]  # value = <Button.DROP_SELECTED_WEAPON: 33>
    JUMP: typing.ClassVar[Button]  # value = <Button.JUMP: 2>
    LAND: typing.ClassVar[Button]  # value = <Button.LAND: 20>
    LOOK_DOWN: typing.ClassVar[Button]  # value = <Button.LOOK_DOWN: 17>
    LOOK_UP: typing.ClassVar[Button]  # value = <Button.LOOK_UP: 16>
    LOOK_UP_DOWN_DELTA: typing.ClassVar[
        Button
    ]  # value = <Button.LOOK_UP_DOWN_DELTA: 38>
    MOVE_BACKWARD: typing.ClassVar[Button]  # value = <Button.MOVE_BACKWARD: 12>
    MOVE_DOWN: typing.ClassVar[Button]  # value = <Button.MOVE_DOWN: 19>
    MOVE_FORWARD: typing.ClassVar[Button]  # value = <Button.MOVE_FORWARD: 13>
    MOVE_FORWARD_BACKWARD_DELTA: typing.ClassVar[
        Button
    ]  # value = <Button.MOVE_FORWARD_BACKWARD_DELTA: 40>
    MOVE_LEFT: typing.ClassVar[Button]  # value = <Button.MOVE_LEFT: 11>
    MOVE_LEFT_RIGHT_DELTA: typing.ClassVar[
        Button
    ]  # value = <Button.MOVE_LEFT_RIGHT_DELTA: 41>
    MOVE_RIGHT: typing.ClassVar[Button]  # value = <Button.MOVE_RIGHT: 10>
    MOVE_UP: typing.ClassVar[Button]  # value = <Button.MOVE_UP: 18>
    MOVE_UP_DOWN_DELTA: typing.ClassVar[
        Button
    ]  # value = <Button.MOVE_UP_DOWN_DELTA: 42>
    RELOAD: typing.ClassVar[Button]  # value = <Button.RELOAD: 6>
    SELECT_NEXT_ITEM: typing.ClassVar[Button]  # value = <Button.SELECT_NEXT_ITEM: 35>
    SELECT_NEXT_WEAPON: typing.ClassVar[
        Button
    ]  # value = <Button.SELECT_NEXT_WEAPON: 31>
    SELECT_PREV_ITEM: typing.ClassVar[Button]  # value = <Button.SELECT_PREV_ITEM: 36>
    SELECT_PREV_WEAPON: typing.ClassVar[
        Button
    ]  # value = <Button.SELECT_PREV_WEAPON: 32>
    SELECT_WEAPON0: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON0: 30>
    SELECT_WEAPON1: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON1: 21>
    SELECT_WEAPON2: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON2: 22>
    SELECT_WEAPON3: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON3: 23>
    SELECT_WEAPON4: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON4: 24>
    SELECT_WEAPON5: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON5: 25>
    SELECT_WEAPON6: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON6: 26>
    SELECT_WEAPON7: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON7: 27>
    SELECT_WEAPON8: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON8: 28>
    SELECT_WEAPON9: typing.ClassVar[Button]  # value = <Button.SELECT_WEAPON9: 29>
    SPEED: typing.ClassVar[Button]  # value = <Button.SPEED: 8>
    STRAFE: typing.ClassVar[Button]  # value = <Button.STRAFE: 9>
    TURN180: typing.ClassVar[Button]  # value = <Button.TURN180: 4>
    TURN_LEFT: typing.ClassVar[Button]  # value = <Button.TURN_LEFT: 15>
    TURN_LEFT_RIGHT_DELTA: typing.ClassVar[
        Button
    ]  # value = <Button.TURN_LEFT_RIGHT_DELTA: 39>
    TURN_RIGHT: typing.ClassVar[Button]  # value = <Button.TURN_RIGHT: 14>
    USE: typing.ClassVar[Button]  # value = <Button.USE: 1>
    ZOOM: typing.ClassVar[Button]  # value = <Button.ZOOM: 7>
    __members__: typing.ClassVar[
        dict[str, Button]
    ]  # value = {'ATTACK': <Button.ATTACK: 0>, 'USE': <Button.USE: 1>, 'JUMP': <Button.JUMP: 2>, 'CROUCH': <Button.CROUCH: 3>, 'TURN180': <Button.TURN180: 4>, 'ALTATTACK': <Button.ALTATTACK: 5>, 'RELOAD': <Button.RELOAD: 6>, 'ZOOM': <Button.ZOOM: 7>, 'SPEED': <Button.SPEED: 8>, 'STRAFE': <Button.STRAFE: 9>, 'MOVE_RIGHT': <Button.MOVE_RIGHT: 10>, 'MOVE_LEFT': <Button.MOVE_LEFT: 11>, 'MOVE_BACKWARD': <Button.MOVE_BACKWARD: 12>, 'MOVE_FORWARD': <Button.MOVE_FORWARD: 13>, 'TURN_RIGHT': <Button.TURN_RIGHT: 14>, 'TURN_LEFT': <Button.TURN_LEFT: 15>, 'LOOK_UP': <Button.LOOK_UP: 16>, 'LOOK_DOWN': <Button.LOOK_DOWN: 17>, 'MOVE_UP': <Button.MOVE_UP: 18>, 'MOVE_DOWN': <Button.MOVE_DOWN: 19>, 'LAND': <Button.LAND: 20>, 'SELECT_WEAPON1': <Button.SELECT_WEAPON1: 21>, 'SELECT_WEAPON2': <Button.SELECT_WEAPON2: 22>, 'SELECT_WEAPON3': <Button.SELECT_WEAPON3: 23>, 'SELECT_WEAPON4': <Button.SELECT_WEAPON4: 24>, 'SELECT_WEAPON5': <Button.SELECT_WEAPON5: 25>, 'SELECT_WEAPON6': <Button.SELECT_WEAPON6: 26>, 'SELECT_WEAPON7': <Button.SELECT_WEAPON7: 27>, 'SELECT_WEAPON8': <Button.SELECT_WEAPON8: 28>, 'SELECT_WEAPON9': <Button.SELECT_WEAPON9: 29>, 'SELECT_WEAPON0': <Button.SELECT_WEAPON0: 30>, 'SELECT_NEXT_WEAPON': <Button.SELECT_NEXT_WEAPON: 31>, 'SELECT_PREV_WEAPON': <Button.SELECT_PREV_WEAPON: 32>, 'DROP_SELECTED_WEAPON': <Button.DROP_SELECTED_WEAPON: 33>, 'ACTIVATE_SELECTED_ITEM': <Button.ACTIVATE_SELECTED_ITEM: 34>, 'SELECT_NEXT_ITEM': <Button.SELECT_NEXT_ITEM: 35>, 'SELECT_PREV_ITEM': <Button.SELECT_PREV_ITEM: 36>, 'DROP_SELECTED_ITEM': <Button.DROP_SELECTED_ITEM: 37>, 'LOOK_UP_DOWN_DELTA': <Button.LOOK_UP_DOWN_DELTA: 38>, 'TURN_LEFT_RIGHT_DELTA': <Button.TURN_LEFT_RIGHT_DELTA: 39>, 'MOVE_FORWARD_BACKWARD_DELTA': <Button.MOVE_FORWARD_BACKWARD_DELTA: 40>, 'MOVE_LEFT_RIGHT_DELTA': <Button.MOVE_LEFT_RIGHT_DELTA: 41>, 'MOVE_UP_DOWN_DELTA': <Button.MOVE_UP_DOWN_DELTA: 42>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class DoomGame:
    """
    DoomGame is the main object of the ViZDoom library, representing a single instance of the Doom game and providing the interface for a single agent/player to interact with the game. The object allows sending actions to the game, getting the game state, etc.
    """

    def __init__(self) -> None: ...
    def add_available_button(self, button: Button, max_value: float = -1) -> None:
        """
        Adds :class:`.Button` type (e.g. ``TURN_LEFT``, ``MOVE_FORWARD``) to available buttons and sets the maximum allowed, absolute value for the specified button.
        If the given button has already been added, it will not be added again, but the maximum value will be overridden.

        Has no effect when the game is running.

        Config key: ``availableButtons``/``available_buttons`` (list of values)
        """

    def add_available_game_variable(self, variable: GameVariable) -> None:
        """
        Adds the specified :class:`.GameVariable` to the list of available game variables (e.g. ``HEALTH``, ``AMMO1``, ``ATTACK_READY``) in the :class:`.GameState` returned by :meth:`get_state` method.

        Has no effect when the game is running.

        Config key: ``availableGameVariables``/``available_game_variables`` (list of values)
        """

    def add_game_args(self, args: str) -> None:
        """
        Adds custom arguments that will be passed to ViZDoom process during initialization.
        It is useful for changing additional game settings.
        Use with caution, as in rare cases it may prevent the library from working properly.

        Config key: ``gameArgs``/``game_args``

        See also:

        - `ZDoom Wiki: Command line parameters <http://zdoom.org/wiki/Command_line_parameters>`_
        - `ZDoom Wiki: CVARs (Console Variables) <http://zdoom.org/wiki/CVARS>`_
        """

    def advance_action(self, tics: int = 1, update_state: bool = True) -> None:
        """
        Processes the specified number of tics, the last action set with :meth:`set_action`
        method will be repeated for each tic. If ``update_state`` argument is set,
        the state will be updated after the last processed tic
        and a new reward will be calculated based on all processed tics since last the last state update.
        To get the new state, use :meth:`get_state` and to get the new reward use :meth:`get_last_reward`.
        """

    def clear_available_buttons(self) -> None:
        """
        Clears all available :class:`.Button`s added so far.

        Has no effect when the game is running.
        """

    def clear_available_game_variables(self) -> None:
        """
        Clears the list of available :class:`.GameVariable` s that are included in the :class:`.GameState` returned by :meth:`get_state` method.

        Has no effect when the game is running.
        """

    def clear_game_args(self) -> None:
        """
        Clears all arguments previously added with :meth:`set_game_args` or/and :meth:`add_game_args` methods.
        """

    def close(self) -> None:
        """
        Closes ViZDoom game instance.
        It is automatically invoked by the destructor.
        The game can be initialized again after being closed.
        """

    def get_armor_reward(self) -> float:
        """
        Returns the reward granted to the player for getting armor points.

        Note: added in 1.3.0
        """

    def get_audio_buffer_size(self) -> int:
        """
        Returns the size of the audio buffer.

        Note: added in 1.1.9.


        See also:

        - :class:`.GameState`
        - `examples/python/audio_buffer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py>`_
        """

    def get_audio_sampling_rate(self) -> int:
        """
        Returns the :class:`.SamplingRate` of the audio buffer.

        See also:

        - :class:`.GameState`
        - `examples/python/audio_buffer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py>`_

        Note: added in 1.1.9.
        """

    def get_available_buttons(self) -> list:
        """
        Returns the list of available :class:`.Button` s,
        that were added with :meth:`set_available_buttons` or/and :meth:`add_available_button` methods.
        """

    def get_available_buttons_size(self) -> int:
        """
        Returns the number of available :class:`.Button` s.
        """

    def get_available_game_variables(self) -> list:
        """
        Returns the list of available :class:`.GameVariable` s,
        that were added with :meth:`set_available_game_variables` or/and :meth:`add_available_game_variable` methods.
        """

    def get_available_game_variables_size(self) -> int:
        """
        Returns the number of available :class:`.GameVariable`.
        It corresponds to taking the size of the list returned by :meth:`get_available_game_variables`.
        """

    def get_button(self, button: Button) -> float:
        """
        Returns the current state of the specified :class:`.Button` (``ATTACK``, ``USE`` etc.).
        """

    def get_button_max_value(self, button: Button) -> float:
        """
        Returns the maximum allowed absolute value for the specified :class:`.Button`.
        """

    def get_damage_made_reward(self) -> float:
        """
        Returns the reward granted to the player for damaging an enemy, proportional to the damage dealt.
        Every point of damage dealt to an enemy will result in a reward equal to the value returned by this method.

        Note: added in 1.3.0
        """

    def get_damage_taken_penalty(self) -> float:
        """
        Returns the penalty for the player when damaged by an enemy, proportional to the damage received.
        Every point of damage taken will result in a penalty equal to the value returned by this method.
        It is equal to negation of value returned by :meth:`get_damage_taken_reward`.

        Note: added in 1.3.0
        """

    def get_damage_taken_reward(self) -> float:
        """
        Returns the reward granted to the player when damaged by an enemy, proportional to the damage received.
        Every point of damage taken will result in a reward equal to the value returned by this method.

        Note: added in 1.3.0
        """

    def get_death_penalty(self) -> float:
        """
        Returns the penalty for the player's death.
        """

    def get_death_reward(self) -> float:
        """
        Returns the reward for the player's death. It is equal to negation of value returned by :meth:`get_death_reward`.

        Note: added in 1.3.0
        """

    def get_episode_start_time(self) -> int:
        """
        Returns the start time (delay) of every episode in tics.
        """

    def get_episode_time(self) -> int:
        """
        Returns number of current episode tic.
        """

    def get_episode_timeout(self) -> int:
        """
        Returns the number of tics after which the episode will be finished.
        """

    def get_frag_reward(self) -> float:
        """
        Returns the reward granted to the player for scoring a frag (killing another player in multiplayer).

        Note: added in 1.3.0
        """

    def get_game_args(self) -> str:
        """
        Returns the additional arguments for ViZDoom process set with :meth:`set_game_args` or/and :meth:`add_game_args` methods.

        Note: added in 1.2.3.
        """

    def get_game_variable(self, variable: GameVariable) -> float:
        """
        Returns the current value of the specified :class:`.GameVariable` (``HEALTH``, ``AMMO1`` etc.).
        The specified game variable does not need to be among available game variables (included in the state).
        It could be used for e.g. shaping. Returns 0 in case of not finding given :class:`.GameVariable`.
        """

    def get_health_reward(self) -> float:
        """
        Returns the reward granted to the player for getting health points.

        Note: added in 1.3.0
        """

    def get_hit_reward(self) -> float:
        """
        Returns the reward granted to the player for hitting (damaging) an enemy.

        Note: added in 1.3.0
        """

    def get_hit_taken_penalty(self) -> float:
        """
        Returns the penalty for the player when hit (damaged) by an enemy.
        The penalty is the same despite the amount of damage taken.
        It is equal to negation of value returned by :meth:`get_hit_taken_reward`.

        Note: added in 1.3.0
        """

    def get_hit_taken_reward(self) -> float:
        """
        Returns the reward granted to the player when hit (damaged) by an enemy.
        The reward is the same despite the amount of damage taken.

        Note: added in 1.3.0
        """

    def get_item_reward(self) -> float:
        """
        Returns the reward granted to the player for picking up an item.

        Note: added in 1.3.0
        """

    def get_kill_reward(self) -> float:
        """
        Returns the reward granted to the player for killing an enemy.

        Note: added in 1.3.0
        """

    def get_last_action(self) -> list:
        """
        Returns the last action performed.
        Each value corresponds to a button added with :meth:`set_available_buttons`
        or/and :meth:`add_available_button` (in order of appearance).
        Most useful in ``SPECTATOR`` mode.
        """

    def get_last_reward(self) -> float:
        """
        Returns a reward granted after the last update of state.
        """

    def get_living_reward(self) -> float:
        """
        Returns the reward granted to the player after every tic.
        """

    def get_map_exit_reward(self) -> float:
        """
        Returns the reward for finishing a map.

        Note: added in 1.3.0
        """

    def get_mode(self) -> Mode:
        """
        Returns the current :class:`.Mode` (``PLAYER``, ``SPECTATOR``, ``ASYNC_PLAYER``, ``ASYNC_SPECTATOR``).
        """

    def get_notifications_buffer_size(self) -> int:
        """
        Returns the size of the notify buffer.

        Note: added in 1.3.0.


        See also:

        - :class:`.GameState`
        - `examples/python/buffers.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py>`_
        """

    def get_screen_channels(self) -> int:
        """
        Returns number of channels in screen buffer and map buffer (depth and labels buffer always have one channel).
        """

    def get_screen_format(self) -> ScreenFormat:
        """
        Returns the format of the screen buffer and the automap buffer.
        """

    def get_screen_height(self) -> int:
        """
        Returns game's screen height - height of screen, depth, labels, and automap buffers.
        """

    def get_screen_pitch(self) -> int:
        """
        Returns size in bytes of one row in screen buffer and map buffer.
        """

    def get_screen_size(self) -> int:
        """
        Returns size in bytes of screen buffer and map buffer.
        """

    def get_screen_width(self) -> int:
        """
        Returns game's screen width - width of screen, depth, labels, and automap buffers.
        """

    def get_secret_reward(self) -> float:
        """
        Returns the reward granted to the player for discovering a secret.

        Note: added in 1.3.0
        """

    def get_seed(self) -> int:
        """
        Returns ViZDoom's seed.
        """

    def get_server_state(self) -> ServerState:
        """
        Returns :class:`.ServerState` object with the current server state.

        Note: added in 1.1.6.
        """

    def get_state(self) -> GameState:
        """
        Returns :class:`.GameState` object with the current game state.
        If the current episode is finished, ``None`` will be returned.

        Note: Changed in 1.1.0
        """

    def get_ticrate(self) -> int:
        """
        Returns current ticrate.

        Note: added in 1.1.0.
        """

    def get_total_reward(self) -> float:
        """
        Returns the sum of all rewards gathered in the current episode.
        """

    def init(self) -> None:
        """
        Initializes ViZDoom game instance and starts a new episode.
        After calling this method, the first state from a new episode will be available.
        Some configuration options cannot be changed after calling this method.
        Init returns ``True`` when the game was started properly and ``False`` otherwise.
        """

    def is_audio_buffer_enabled(self) -> bool:
        """
        Returns ``True`` if the audio buffer is enabled.

        Note: added in 1.1.9.
        """

    def is_automap_buffer_enabled(self) -> bool:
        """
        Returns ``True`` if the automap buffer is enabled.

        Note: added in 1.1.0.
        """

    def is_depth_buffer_enabled(self) -> bool:
        """
        Returns ``True`` if the depth buffer is enabled.
        """

    def is_episode_finished(self) -> bool:
        """
        Returns ``True`` if the current episode is in the terminal state (is finished).
        :meth:`make_action` and :meth:`advance_action` methods
        will take no effect after this point (unless :meth:`new_episode` method is called).
        """

    def is_episode_timeout_reached(self) -> bool:
        """
        Returns ``True`` if the current episode is in the terminal state due to exceeding the time limit (timeout)
        set with :meth:`set_episode_timeout`` method or via ``+timelimit` parameter.
        """

    def is_labels_buffer_enabled(self) -> bool:
        """
        Returns ``True`` if the labels buffer is enabled.

        Note: added in 1.1.0.
        """

    def is_multiplayer_game(self) -> bool:
        """
        Returns ``True`` if the game is in multiplayer mode.

        See also:

        - `examples/python/multiple_instances.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/multiple_instances.py>`_
        - `examples/python/cig_multiplayer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/cig_multiplayer.py>`_
        - `examples/python/cig_multiplayer_host.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/cig_multiplayer_host.py>`_

        Note: added in 1.1.2.
        """

    def is_new_episode(self) -> bool:
        """
        Returns ``True`` if the current episode is in the initial state - the first state, no actions were performed yet.
        """

    def is_notifications_buffer_enabled(self) -> bool:
        """
        Returns ``True`` if the notify buffer is enabled.

        Note: added in 1.3.0.
        """

    def is_objects_info_enabled(self) -> bool:
        """
        Returns ``True`` if the objects information is enabled.

        Note: added in 1.1.8.
        """

    def is_player_dead(self) -> bool:
        """
        Returns ``True`` if the player is dead.
        In singleplayer, the player's death is equivalent to the end of the episode.
        In multiplayer, when the player is dead :meth:`respawn_player` method can be called.
        """

    def is_recording_episode(self) -> bool:
        """
        Returns ``True`` if the game is in recording mode.

        Note: added in 1.1.5.
        """

    def is_replaying_episode(self) -> bool:
        """
        Returns ``True`` if the game is in replay mode.

        Note: added in 1.1.5.
        """

    def is_running(self) -> bool:
        """
        Returns ``True`` if the controlled game instance is running.
        """

    def is_sectors_info_enabled(self) -> bool:
        """
        Returns ``True`` if the information about sectors is enabled.

        Note: added in 1.1.8.
        """

    def load(self, file_path: str) -> None:
        """
        Loads a game's internal state from the file using ZDoom load game functionality.
        A new state is available after loading.
        Loading the game state does not reset the current episode state,
        tic counter/time and total reward state keep their values.

        Note: added in 1.1.9.
        """

    def load_config(self, config: str) -> bool:
        """
        Loads configuration (resolution, available buttons, game variables etc.) from a configuration file.
        In case of multiple invocations, older configurations will be overwritten by the recent ones.
        Overwriting does not involve resetting to default values. Thus only overlapping parameters will be changed.
        The method returns ``True`` if the whole configuration file was correctly read and applied,
        `False` if the file contained errors.

        If the file relative path is given, it will be searched for in the following order: ``<current directory>``, ``<current directory>/scenarios/``, ``<ViZDoom library location>/scenarios/``.
        """

    def make_action(self, action: typing.Any, tics: int = 1) -> float:
        """
        This method combines functionality of :meth:`set_action`, :meth:`advance_action`,
        and :meth:`get_last_reward` called in this sequance.
        Sets the player's action for all the next tics (the same action will be repeated for each tic),
        processes the specified number of tics, updates the state and calculates a new reward from all processed tics, which is returned.
        """

    def new_episode(self, recording_file_path: str = "") -> None:
        """
        Initializes a new episode. The state of an environment is completely restarted (all variables and rewards are reset to their initial values).
        After calling this method, the first state from the new episode will be available.
        If the ``recording_file_path`` argument is not empty, the new episode will be recorded to this file (as a Doom lump).

        In a multiplayer game, the host can call this method to finish the game.
        Then the rest of the players must also call this method to start a new episode.

        Note: Changed in 1.1.0
        """

    def replay_episode(self, file_path: str, player: int = 0) -> None:
        """
        Replays the recorded episode from the given file using the perspective of the specified player.
        Players are numbered from 1, If ``player`` argument is equal to 0,
        the episode will be replayed using the perspective of the default player in the recording file.
        After calling this method, the first state from the replay will be available.
        All rewards, variables, and states are available when replaying the episode.

        See also:

        - `examples/python/record_episodes.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/record_episodes.py>`_
        - `examples/python/record_multiplayer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/record_multiplayer.py>`_

        Note: added in 1.1.0.
        """

    def respawn_player(self) -> None:
        """
        This method respawns the player after death in multiplayer mode.
        After calling this method, the first state after the respawn will be available.

        See also:

        - :meth:`is_multiplayer_game`
        """

    def save(self, file_path: str) -> None:
        """
        Saves a game's internal state to the file using ZDoom save game functionality.

        Note: added in 1.1.9.
        """

    def send_game_command(self, cmd: str) -> None:
        """
        Sends the command to Doom console. It can be used for controlling the game, changing settings, cheats, etc.
        Some commands will be blocked in some modes.

        See also:

        - `ZDoom Wiki: Console <http://zdoom.org/wiki/Console>`_
        - `ZDoom Wiki: CVARs (console variables) <https://zdoom.org/wiki/CVARs>`_
        - `ZDoom Wiki: CCMD (console commands) <https://zdoom.org/wiki/CCMDs>`_
        """

    def set_action(self, action: typing.Any) -> None:
        """
        Sets the player's action for the following tics until the method is called again with new action.
        Each value corresponds to a button previously specified
        with :meth:`add_available_button`, or :meth:`set_available_buttons` methods,
        or in the configuration file (in order of appearance).
        """

    def set_armor_reward(self, armor_reward: float) -> None:
        """
        Sets the reward granted to the player for getting armor points. A negative value is also allowed.

        Default value: 0

        Config key: ``armorReward``/``armor_reward``

        Note: added in 1.3.0
        """

    def set_audio_buffer_enabled(self, audio_buffer: bool) -> None:
        """
        Enables rendering of the audio buffer, it will be available in the state.
        The audio buffer will contain audio from the number of the last tics specified by :meth:`set_audio_buffer_size` method.
        Sampling rate can be set with :meth:`set_audio_sampling_rate` method.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``audioBufferEnabled``/``audio_buffer_enabled``

        See also:

        - :class:`.GameState`
        - :class:`.SamplingRate`
        - `examples/python/audio_buffer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py>`_

        Note: added in 1.1.9.
        """

    def set_audio_buffer_size(self, tics: int) -> None:
        """
        Sets the size/length of the audio buffer. The size is defined by a number of logic tics.
        After each action audio buffer will contain audio from the specified number of the last processed tics.
        Doom uses 35 ticks per second.

        Default value: 1

        Has no effect when the game is running.

        Config key: ``audioBufferSize``/``audio_buffer_size``

        See also:

        - :class:`.GameState`
        - `examples/python/audio_buffer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py>`_

        Note: added in 1.1.9.
        """

    def set_audio_sampling_rate(self, sampling_rate: SamplingRate) -> None:
        """
        Sets the :class:`.SamplingRate` of the audio buffer.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``audioSamplingRate``/``audio_sampling_rate``

        See also:

        - :class:`.GameState`
        - `examples/python/audio_buffer.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/audio_buffer.py>`_

        Note: added in 1.1.9.
        """

    def set_automap_buffer_enabled(self, automap_buffer: bool) -> None:
        """
        Enables rendering of the automap buffer, it will be available in the state.
        The buffer always has the same resolution as the screen buffer.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``automapBufferEnabled``/``automap_buffer_enabled``

        See also:

        - :class:`.GameState`
        - `examples/python/buffers.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py>`_,

        Note: added in 1.1.0.
        """

    def set_automap_mode(self, mode: AutomapMode) -> None:
        """
        Sets the :class:`.AutomapMode` (``NORMAL``, ``WHOLE``, ``OBJECTS``, ``OBJECTS_WITH_SIZE``),
        which determines what will be visible on it.

        Default value: ``NORMAL``

        Config key: ``automapMode``/``automap_mode``

        Note: added in 1.1.0.
        """

    def set_automap_render_textures(self, textures: bool) -> None:
        """
        Determine if the automap will be textured, showing the floor textures.

        Default value: ``True``

        Config key: ``automapRenderTextures``/``automap_render_textures``

        Note: added in 1.1.0.
        """

    def set_automap_rotate(self, rotate: bool) -> None:
        """
        Determine if the automap will be rotating with the player.
        If ``False``, north always will be at the top of the buffer.

        Default value: ``False``

        Config key: ``automapRotate``/``automap_rotate``

        Note: added in 1.1.0.
        """

    def set_available_buttons(self, buttons: list) -> None:
        """
        Sets given list of :class:`.Button` s (e.g. ``TURN_LEFT``, ``MOVE_FORWARD``) as available buttons.

        Has no effect when the game is running.

        Default value: `[]` (empty vector/list, no buttons).

        Config key: ``availableButtons``/``available_buttons`` (list of values)
        """

    def set_available_game_variables(self, variables: list) -> None:
        """
        Sets list of :class:`.GameVariable` s as available game variables in the :class:`.GameState` returned by :meth:`get_state` method.

        Has no effect when the game is running.

        Default value: `[]` (empty vector/list, no game variables).

        Config key: ``availableGameVariables``/``available_game_variables`` (list of values)
        """

    def set_button_max_value(self, button: Button, max_value: float) -> None:
        """
        Sets the maximum allowed absolute value for the specified :class:`.Button`.
        Setting the maximum value to 0 results in no constraint at all (infinity).
        This method makes sense only for delta buttons.
        The constraints limit applies in all Modes.

        Default value: 0 (no constraint, infinity).

        Has no effect when the game is running.
        """

    def set_console_enabled(self, console: bool) -> None:
        """
        Determines if ViZDoom's console output will be enabled.

        Default value: ``False``

        Config key: ``consoleEnabled``/``console_enabled``
        """

    def set_damage_made_reward(self, damage_made_reward: float) -> None:
        """
        Sets the reward granted to the player for damaging an enemy, proportional to the damage dealt.
        Every point of damage dealt to an enemy will result in a reward equal to the value returned by this method.
        A negative value is also allowed.


        Default value: 0

        Config key: ``damageMadeReward``/``damage_made_reward``

        Note: added in 1.3.0
        """

    def set_damage_taken_penalty(self, damage_taken_penalty: float) -> None:
        """
        Sets a penalty for the player when damaged by an enemy, proportional to the damage received.
        Every point of damage taken will result in a penalty equal to the set value.
        Note that in case of a negative value, the player will be rewarded upon receiving damage.

        Default value: 0

        Config key: ``damageTakenPenalty``/``damage_taken_penalty``

        Note: added in 1.3.0
        """

    def set_damage_taken_reward(self, damage_taken_reward: float) -> None:
        """
        Sets the reward granted to the player when damaged by an enemy, proportional to the damage received.
        Every point of damage taken will result in a reward equal to the set value.
        A negative value is also allowed.

        Default value: 0

        Config key: ``damageTakenReward``/``damage_taken_reward``

        Note: added in 1.3.0
        """

    def set_death_penalty(self, death_penalty: float) -> None:
        """
        Sets a penalty for the player's death. Note that in case of a negative value, the player will be rewarded upon dying.

        Default value: 0

        Config key: ``deathPenalty``/``death_penalty``
        """

    def set_death_reward(self, death_reward: float) -> None:
        """
        Sets a reward for the player's death. A negative value is also allowed.

        Default value: 0

        Config key: ``deathReward``/``death_reward``

        Note: added in 1.3.0
        """

    def set_depth_buffer_enabled(self, depth_buffer: bool) -> None:
        """
        Enables rendering of the depth buffer, it will be available in the state.
        The buffer always has the same resolution as the screen buffer.
        Depth buffer will contain noise if ``viz_nocheat`` flag is enabled.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``depthBufferEnabled``/``depth_buffer_enabled``

        See also:

        - :class:`.GameState`
        - `examples/python/buffers.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py>`_

        Note: added in 1.1.0.
        """

    def set_doom_config_path(self, button: str) -> None:
        """
        Sets the path for ZDoom's configuration file.
        The file is responsible for the configuration of the ZDoom engine itself.
        If it does not exist, it will be created after the ``vizdoom`` executable is run.
        This method is not needed for most of the tasks and is added for the convenience of users with hacking tendencies.

        Default value: ``""``, if left empty ``"_vizdoom.ini"`` will be used.

        Config key: ``DoomConfigPath``/``doom_config_path``
        """

    def set_doom_game_path(self, button: str) -> None:
        """
        Sets the path to the Doom engine based game file (wad format).
        If not used DoomGame will look for doom2.wad and freedoom2.wad (in that order) in the directory of ViZDoom's installation (where vizdoom library/pyd is).

        Default value: ``<ViZDoom library location>/<doom2.wad, doom.wad, freedoom2.wad, or freedoom.wad - in this order>``

        Config key: ``DoomGamePath``/``doom_game_path``
        """

    def set_doom_map(self, button: str) -> None:
        """
        Sets the map name to be used.

        Default value: ``"map01"``, if set to empty ``"map01"`` will be used.

        Config key: ``DoomMap``/``doom_map``
        """

    def set_doom_scenario_path(self, button: str) -> None:
        """
        Sets the path to an additional scenario file (wad format).
        If not provided, the default Doom single-player maps will be loaded.

        Default value: ``""``

        Config key: ``DoomScenarioPath``/``doom_scenario_path``
        """

    def set_doom_skill(self, button: int) -> None:
        """
        Sets Doom game difficulty level, which is called skill in Doom.
        The higher the skill, the harder the game becomes.
        Skill level affects monsters' aggressiveness, monsters' speed, weapon damage, ammunition quantities, etc.
        Takes effect from the next episode.

        - 1 - VERY EASY, “I'm Too Young to Die” in Doom.
        - 2 - EASY, “Hey, Not Too Rough" in Doom.
        - 3 - NORMAL, “Hurt Me Plenty” in Doom.
        - 4 - HARD, “Ultra-Violence” in Doom.
        - 5 - VERY HARD, “Nightmare!” in Doom.

        Default value: 3

        Config key: ``DoomSkill``/``doom_skill``
        """

    def set_episode_start_time(self, start_time: int) -> None:
        """
        Sets the start time (delay) of every episode in tics.
        Every episode will effectively start (from the user's perspective) after the provided number of tics.

        Default value: 1

        Config key: ``episodeStartTime``/``episode_start_time``
        """

    def set_episode_timeout(self, timeout: int) -> None:
        """
        Sets the number of tics after which the episode will be finished. 0 will result in no timeout.

        Default value: 0

        Config key: ``episodeTimeout``/``episode_timeout``
        """

    def set_frag_reward(self, frag_reward: float) -> None:
        """
        Sets the reward granted to the player for scoring a frag. A negative value is also allowed.

        Default value: 0

        Config key: ``fragReward``/``frag_reward``

        Note: added in 1.3.0
        """

    def set_game_args(self, args: str) -> None:
        """
        Sets custom arguments that will be passed to ViZDoom process during initialization.
        It is useful for changing additional game settings.
        Use with caution, as in rare cases it may prevent the library from working properly.
        Using this method is equivalent to first calling :meth:`clear_game_args` and then :meth:`add_game_args`.

        Default value: ``""`` (empty string, no additional arguments).

        Config key: ``gameArgs``/``game_args``

        See also:

        - `ZDoom Wiki: Command line parameters <http://zdoom.org/wiki/Command_line_parameters>`_
        - `ZDoom Wiki: CVARs (Console Variables) <http://zdoom.org/wiki/CVARS>`_

        Note: added in 1.2.3.
        """

    def set_health_reward(self, health_reward: float) -> None:
        """
        Sets the reward granted to the player for getting health points. A negative value is also allowed.

        Default value: 0

        Config key: ``healthReward``/``health_reward``

        Note: added in 1.3.0
        """

    def set_hit_reward(self, hit_reward: float) -> None:
        """
        Sets the reward granted to the player for hitting (damaging) an enemy.
        The reward is the same despite the amount of damage dealt.
        A negative value is also allowed.

        Default value: 0

        Config key: ``hitReward``/``hit_reward``

        Note: added in 1.3.0
        """

    def set_hit_taken_penalty(self, hit_taken_penalty: float) -> None:
        """
        Sets a penalty for the player when hit (damaged) by an enemy.
        The penalty is the same despite the amount of damage taken.
        Note that in case of a negative value, the player will be rewarded upon being hit.

        Default value: 0

        Config key: ``hitTakenPenalty``/``hit_taken_penalty``

        Note: added in 1.3.0
        """

    def set_hit_taken_reward(self, hit_taken_reward: float) -> None:
        """
        Sets the reward granted to the player when hit (damaged) by an enemy.
        The reward is the same despite the amount of damage taken.
        A negative value is also allowed.

        Default value: 0

        Config key: ``hitTakenReward``/``hit_taken_reward``

        Note: added in 1.3.0
        """

    def set_item_reward(self, item_reward: float) -> None:
        """
        Sets the reward granted to the player for picking up an item. A negative value is also allowed.

        Default value: 0

        Config key: ``itemReward``/``item_reward``

        Note: added in 1.3.0
        """

    def set_kill_reward(self, kill_reward: float) -> None:
        """
        Sets the reward granted to the player for killing an enemy. A negative value is also allowed.

        Default value: 0

        Config key: ``killReward``/``kill_reward``

        Note: added in 1.3.0
        """

    def set_labels_buffer_enabled(self, labels_buffer: bool) -> None:
        """
        Enables rendering of the labels buffer, it will be available in the state with the vector of :class:`.Label` s.
        The buffer always has the same resolution as the screen buffer.
        LabelsBuffer will contain noise if ``viz_nocheat`` is enabled.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``labelsBufferEnabled``/``labels_buffer_enabled``

        See also:

        - :class:`.GameState`
        - `examples/python/labels.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/labels.py>`_
        - `examples/python/buffers.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py>`_

        Note: added in 1.1.0.
        """

    def set_living_reward(self, living_reward: float) -> None:
        """
        Sets the reward granted to the player after every tic. A negative value is also allowed.

        Default value: 0

        Config key: ``livingReward``/``living_reward``
        """

    def set_map_exit_reward(self, map_exit_reward: float) -> None:
        """
        Sets a reward for finishing a map (finding an exit or succeeding in other programmed objective). A negative value is also allowed.

        Default value: 0

        Config key: ``mapExitReward``/``map_exit_reward``

        Note: added in 1.3.0
        """

    def set_mode(self, mode: Mode) -> None:
        """
        Sets the :class:`.Mode` (``PLAYER``, ``SPECTATOR``, ``ASYNC_PLAYER``, ``ASYNC_SPECTATOR``) in which the game will be running.

        Default value: ``PLAYER``.

        Has no effect when the game is running.

        Config key: ``mode``
        """

    def set_notifications_buffer_enabled(self, notifications_buffer: bool) -> None:
        """
        Enables notification buffer, it will be available in the state.
        The notification buffer will contain text notifications from the number of the last tics specified by :meth:`set_notifications_buffer_size` method.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``notificationsBufferEnabled``/``notifications_buffer_enabled``

        See also:

        - :class:`.GameState`
        - `examples/python/buffers.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py>`_

        Note: added in 1.3.0.
        """

    def set_notifications_buffer_size(self, tics: int) -> None:
        """
        Sets the size of the notify buffer. The size is defined by a number of logic tics.
        After each action notify buffer will contain text notifications from the specified number of the last processed tics.
        Doom uses 35 ticks per second.

        Default value: 1

        Has no effect when the game is running.

        Config key: ``notificationsBufferSize``/``notifications_buffer_size``

        See also:

        - :class:`.GameState`
        - `examples/python/buffers.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/buffers.py>`_

        Note: added in 1.3.0.
        """

    def set_objects_info_enabled(self, objects_info: bool) -> None:
        """
        Enables information about all :class:`.Object` s present in the current episode/level.
        It will be available in the state.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``objectsInfoEnabled``/``objects_info_enabled``

        See also:

        - :class:`.GameState`
        - `examples/python/objects_and_sectors.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py>`_,

        Note: added in 1.1.8.
        """

    def set_render_all_frames(self, all_frames: bool) -> None:
        """
        Determine if all frames between states will be rendered (when skip greater than 1 is used).
        Allows smooth preview but can reduce performance.
        It only makes sense to use it if the window is visible.

        Default value: ``False``

        Config key: ``renderAllFrames``/``render_all_frames``

        See also:

        - :meth:`set_window_visible`

        Note: added in 1.1.3.
        """

    def set_render_corpses(self, bodies: bool) -> None:
        """
        Determine if actors' corpses will be rendered in the game.

        Default value: ``True``

        Config key: ``renderCorpses``/``render_corpses``

        Note: added in 1.1.0.
        """

    def set_render_crosshair(self, crosshair: bool) -> None:
        """
        Determine if the crosshair will be rendered in the game.

        Default value: ``False``

        Config key: ``renderCrosshair``/``render_crosshair``
        """

    def set_render_decals(self, decals: bool) -> None:
        """
        Determine if the decals (marks on the walls) will be rendered in the game.

        Default value: ``True``

        Config key: ``renderDecals``/``render_decals``
        """

    def set_render_effects_sprites(self, sprites: bool) -> None:
        """
        Determine if some effects sprites (gun puffs, blood splats etc.) will be rendered in the game.

        Default value: ``True``

        Config key: ``renderEffectsSprites``/``render_effects_sprites``

        Note: added in 1.1.0.
        """

    def set_render_hud(self, hud: bool) -> None:
        """
        Determine if the hud will be rendered in the game.

        Default value: ``False``

        Config key: ``renderHud``/``render_hud``
        """

    def set_render_messages(self, messages: bool) -> None:
        """
        Determine if in-game messages (information about pickups, kills, etc.) will be rendered in the game.

        Default value: ``False``

        Config key: ``renderMessages``/``render_messages``

        Note: added in 1.1.0.
        """

    def set_render_minimal_hud(self, min_hud: bool) -> None:
        """
        Determine if the minimalistic version of the hud will be rendered instead of the full hud.

        Default value: ``False``

        Config key: ``renderMinimalHud``/``render_minimal_hud``

        Note: added in 1.1.0.
        """

    def set_render_particles(self, particles: bool) -> None:
        """
        Determine if the particles will be rendered in the game.

        Default value: ``True``

        Config key: ``renderParticles``/``render_particles``
        """

    def set_render_screen_flashes(self, flashes: bool) -> None:
        """
        Determine if the screen flash effect upon taking damage or picking up items will be rendered in the game.

        Default value: ``True``

        Config key: ``renderScreenFlashes``/``render_screen_flashes``

        Note: added in 1.1.0.
        """

    def set_render_weapon(self, weapon: bool) -> None:
        """
        Determine if the weapon held by the player will be rendered in the game.

        Default value: ``True``

        Config key: ``renderWeapon``/``render_weapon``
        """

    def set_screen_format(self, format: ScreenFormat) -> None:
        """
        Sets the format of the screen buffer and the automap buffer.
        Supported formats are defined in :class:`.ScreenFormat` enumeration type (e.g. ``CRCGCB``, ``RGB24``, ``GRAY8``).
        The format change affects only the buffers, so it will not have any effect on the content of ViZDoom's display window.

        Default value: ``CRCGCB``

        Has no effect when the game is running.

        Config key: ``screenFormat``/``screen_format``
        """

    def set_screen_resolution(self, resolution: ScreenResolution) -> None:
        """
        Sets the screen resolution and additional buffers (depth, labels, and automap). ZDoom engine supports only specific resolutions.
        Supported resolutions are part of :class:`.ScreenResolution` enumeration (e.g., ``RES_320X240``, ``RES_640X480``, ``RES_1920X1080``).
        The buffers, as well as the content of ViZDoom's display window, will be affected.

        Default value: ``RES_320X240``

        Has no effect when the game is running.

        Config key: ``screenResolution``/``screen_resolution``
        """

    def set_secret_reward(self, secret_reward: float) -> None:
        """
        Sets the reward granted to the player for discovering a secret. A negative value is also allowed.

        Default value: 0

        Config key: ``secretReward``/``secret_reward``

        Note: added in 1.3.0
        """

    def set_sectors_info_enabled(self, sectors_info: bool) -> None:
        """
        Enables information about all :class:`.Sector` s (map layout) present in the current episode/level.
        It will be available in the state.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``sectorsInfoEnabled``/``sectors_info_enabled``

        See also:

        - :class:`.GameState`
        - `examples/python/objects_and_sectors.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/objects_and_sectors.py>`_

        Note: added in 1.1.8.
        """

    def set_seed(self, seed: int) -> None:
        """
        Sets the seed of ViZDoom's RNG that generates seeds (initial state) for episodes.

        Default value: randomized in constructor

        Config key: ``seed``

        See also:

        - `examples/python/seed.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/seed.py>`_
        """

    def set_sound_enabled(self, sound: bool) -> None:
        """
        Determines if ViZDoom's sound will be played.

        Default value: ``False``

        Config key: ``soundEnabled``/``sound_enabled``
        """

    def set_ticrate(self, button: int) -> None:
        """
        Sets the ticrate for ASNYC Modes - number of logic tics executed per second.
        The default Doom ticrate is 35. This value will play a game at normal speed.

        Default value: 35 (default Doom ticrate).

        Has no effect when the game is running.

        Config key: ``ticrate``

        See also:

        - `examples/python/ticrate.py <https://github.com/Farama-Foundation/ViZDoom/tree/master/examples/python/ticrate.py>`_

        Note: added in 1.1.0.
        """

    def set_vizdoom_path(self, button: str) -> None:
        """
        Sets the path to the ViZDoom engine executable vizdoom.

        Default value: ``<ViZDoom library location>/<vizdoom or vizdoom.exe on Windows>``.

        Config key: ``ViZDoomPath``/``vizdoom_path``
        """

    def set_window_visible(self, visiblity: bool) -> None:
        """
        Determines if ViZDoom's window will be visible.
        ViZDoom with window disabled can be used on Linux systems without X Server.

        Default value: ``False``

        Has no effect when the game is running.

        Config key: ``windowVisible``/``window_visible``
        """

class FileDoesNotExistException(Exception):
    pass

class GameState:
    """
    Contains the state of the game including screen buffer, game variables, and world geometry, available information depand on the configuration of the game instance.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def audio_buffer(self) -> typing.Optional[NDArray]: ...
    @property
    def automap_buffer(self) -> typing.Optional[NDArray]: ...
    @property
    def depth_buffer(self) -> typing.Optional[NDArray]: ...
    @property
    def game_variables(self) -> typing.Optional[NDArray]: ...
    @property
    def labels(self) -> typing.Any: ...
    @property
    def labels_buffer(self) -> typing.Optional[NDArray]: ...
    @property
    def notifications_buffer(self) -> typing.Any: ...
    @property
    def number(self) -> int: ...
    @property
    def objects(self) -> typing.Any: ...
    @property
    def screen_buffer(self) -> NDArray: ...
    @property
    def sectors(self) -> typing.Any: ...
    @property
    def tic(self) -> int: ...

class GameVariable:
    """
    Defines available game variables that can be accessed to get information about the game state.

    Members:

      KILLCOUNT

      ITEMCOUNT

      SECRETCOUNT

      FRAGCOUNT

      DEATHCOUNT

      HITCOUNT

      HITS_TAKEN

      DAMAGECOUNT

      DAMAGE_TAKEN

      HEALTH

      ARMOR

      DEAD

      ON_GROUND

      ATTACK_READY

      ALTATTACK_READY

      SELECTED_WEAPON

      SELECTED_WEAPON_AMMO

      AMMO1

      AMMO2

      AMMO3

      AMMO4

      AMMO5

      AMMO6

      AMMO7

      AMMO8

      AMMO9

      AMMO0

      WEAPON1

      WEAPON2

      WEAPON3

      WEAPON4

      WEAPON5

      WEAPON6

      WEAPON7

      WEAPON8

      WEAPON9

      WEAPON0

      POSITION_X

      POSITION_Y

      POSITION_Z

      ANGLE

      PITCH

      ROLL

      VIEW_HEIGHT

      VELOCITY_X

      VELOCITY_Y

      VELOCITY_Z

      CAMERA_POSITION_X

      CAMERA_POSITION_Y

      CAMERA_POSITION_Z

      CAMERA_ANGLE

      CAMERA_PITCH

      CAMERA_ROLL

      CAMERA_FOV

      USER1

      USER2

      USER3

      USER4

      USER5

      USER6

      USER7

      USER8

      USER9

      USER10

      USER11

      USER12

      USER13

      USER14

      USER15

      USER16

      USER17

      USER18

      USER19

      USER20

      USER21

      USER22

      USER23

      USER24

      USER25

      USER26

      USER27

      USER28

      USER29

      USER30

      USER31

      USER32

      USER33

      USER34

      USER35

      USER36

      USER37

      USER38

      USER39

      USER40

      USER41

      USER42

      USER43

      USER44

      USER45

      USER46

      USER47

      USER48

      USER49

      USER50

      USER51

      USER52

      USER53

      USER54

      USER55

      USER56

      USER57

      USER58

      USER59

      USER60

      PLAYER_NUMBER

      PLAYER_COUNT

      PLAYER1_FRAGCOUNT

      PLAYER2_FRAGCOUNT

      PLAYER3_FRAGCOUNT

      PLAYER4_FRAGCOUNT

      PLAYER5_FRAGCOUNT

      PLAYER6_FRAGCOUNT

      PLAYER7_FRAGCOUNT

      PLAYER8_FRAGCOUNT

      PLAYER9_FRAGCOUNT

      PLAYER10_FRAGCOUNT

      PLAYER11_FRAGCOUNT

      PLAYER12_FRAGCOUNT

      PLAYER13_FRAGCOUNT

      PLAYER14_FRAGCOUNT

      PLAYER15_FRAGCOUNT

      PLAYER16_FRAGCOUNT
    """

    ALTATTACK_READY: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.ALTATTACK_READY: 14>
    AMMO0: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO0: 17>
    AMMO1: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO1: 18>
    AMMO2: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO2: 19>
    AMMO3: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO3: 20>
    AMMO4: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO4: 21>
    AMMO5: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO5: 22>
    AMMO6: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO6: 23>
    AMMO7: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO7: 24>
    AMMO8: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO8: 25>
    AMMO9: typing.ClassVar[GameVariable]  # value = <GameVariable.AMMO9: 26>
    ANGLE: typing.ClassVar[GameVariable]  # value = <GameVariable.ANGLE: 40>
    ARMOR: typing.ClassVar[GameVariable]  # value = <GameVariable.ARMOR: 10>
    ATTACK_READY: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.ATTACK_READY: 13>
    CAMERA_ANGLE: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.CAMERA_ANGLE: 50>
    CAMERA_FOV: typing.ClassVar[GameVariable]  # value = <GameVariable.CAMERA_FOV: 53>
    CAMERA_PITCH: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.CAMERA_PITCH: 51>
    CAMERA_POSITION_X: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.CAMERA_POSITION_X: 47>
    CAMERA_POSITION_Y: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.CAMERA_POSITION_Y: 48>
    CAMERA_POSITION_Z: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.CAMERA_POSITION_Z: 49>
    CAMERA_ROLL: typing.ClassVar[GameVariable]  # value = <GameVariable.CAMERA_ROLL: 52>
    DAMAGECOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.DAMAGECOUNT: 7>
    DAMAGE_TAKEN: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.DAMAGE_TAKEN: 8>
    DEAD: typing.ClassVar[GameVariable]  # value = <GameVariable.DEAD: 11>
    DEATHCOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.DEATHCOUNT: 4>
    FRAGCOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.FRAGCOUNT: 3>
    HEALTH: typing.ClassVar[GameVariable]  # value = <GameVariable.HEALTH: 9>
    HITCOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.HITCOUNT: 5>
    HITS_TAKEN: typing.ClassVar[GameVariable]  # value = <GameVariable.HITS_TAKEN: 6>
    ITEMCOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.ITEMCOUNT: 1>
    KILLCOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.KILLCOUNT: 0>
    ON_GROUND: typing.ClassVar[GameVariable]  # value = <GameVariable.ON_GROUND: 12>
    PITCH: typing.ClassVar[GameVariable]  # value = <GameVariable.PITCH: 41>
    PLAYER10_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER10_FRAGCOUNT: 65>
    PLAYER11_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER11_FRAGCOUNT: 66>
    PLAYER12_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER12_FRAGCOUNT: 67>
    PLAYER13_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER13_FRAGCOUNT: 68>
    PLAYER14_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER14_FRAGCOUNT: 69>
    PLAYER15_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER15_FRAGCOUNT: 70>
    PLAYER16_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER16_FRAGCOUNT: 71>
    PLAYER1_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER1_FRAGCOUNT: 56>
    PLAYER2_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER2_FRAGCOUNT: 57>
    PLAYER3_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER3_FRAGCOUNT: 58>
    PLAYER4_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER4_FRAGCOUNT: 59>
    PLAYER5_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER5_FRAGCOUNT: 60>
    PLAYER6_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER6_FRAGCOUNT: 61>
    PLAYER7_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER7_FRAGCOUNT: 62>
    PLAYER8_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER8_FRAGCOUNT: 63>
    PLAYER9_FRAGCOUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER9_FRAGCOUNT: 64>
    PLAYER_COUNT: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER_COUNT: 55>
    PLAYER_NUMBER: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.PLAYER_NUMBER: 54>
    POSITION_X: typing.ClassVar[GameVariable]  # value = <GameVariable.POSITION_X: 37>
    POSITION_Y: typing.ClassVar[GameVariable]  # value = <GameVariable.POSITION_Y: 38>
    POSITION_Z: typing.ClassVar[GameVariable]  # value = <GameVariable.POSITION_Z: 39>
    ROLL: typing.ClassVar[GameVariable]  # value = <GameVariable.ROLL: 42>
    SECRETCOUNT: typing.ClassVar[GameVariable]  # value = <GameVariable.SECRETCOUNT: 2>
    SELECTED_WEAPON: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.SELECTED_WEAPON: 15>
    SELECTED_WEAPON_AMMO: typing.ClassVar[
        GameVariable
    ]  # value = <GameVariable.SELECTED_WEAPON_AMMO: 16>
    USER1: typing.ClassVar[GameVariable]  # value = <GameVariable.USER1: 72>
    USER10: typing.ClassVar[GameVariable]  # value = <GameVariable.USER10: 81>
    USER11: typing.ClassVar[GameVariable]  # value = <GameVariable.USER11: 82>
    USER12: typing.ClassVar[GameVariable]  # value = <GameVariable.USER12: 83>
    USER13: typing.ClassVar[GameVariable]  # value = <GameVariable.USER13: 84>
    USER14: typing.ClassVar[GameVariable]  # value = <GameVariable.USER14: 85>
    USER15: typing.ClassVar[GameVariable]  # value = <GameVariable.USER15: 86>
    USER16: typing.ClassVar[GameVariable]  # value = <GameVariable.USER16: 87>
    USER17: typing.ClassVar[GameVariable]  # value = <GameVariable.USER17: 88>
    USER18: typing.ClassVar[GameVariable]  # value = <GameVariable.USER18: 89>
    USER19: typing.ClassVar[GameVariable]  # value = <GameVariable.USER19: 90>
    USER2: typing.ClassVar[GameVariable]  # value = <GameVariable.USER2: 73>
    USER20: typing.ClassVar[GameVariable]  # value = <GameVariable.USER20: 91>
    USER21: typing.ClassVar[GameVariable]  # value = <GameVariable.USER21: 92>
    USER22: typing.ClassVar[GameVariable]  # value = <GameVariable.USER22: 93>
    USER23: typing.ClassVar[GameVariable]  # value = <GameVariable.USER23: 94>
    USER24: typing.ClassVar[GameVariable]  # value = <GameVariable.USER24: 95>
    USER25: typing.ClassVar[GameVariable]  # value = <GameVariable.USER25: 96>
    USER26: typing.ClassVar[GameVariable]  # value = <GameVariable.USER26: 97>
    USER27: typing.ClassVar[GameVariable]  # value = <GameVariable.USER27: 98>
    USER28: typing.ClassVar[GameVariable]  # value = <GameVariable.USER28: 99>
    USER29: typing.ClassVar[GameVariable]  # value = <GameVariable.USER29: 100>
    USER3: typing.ClassVar[GameVariable]  # value = <GameVariable.USER3: 74>
    USER30: typing.ClassVar[GameVariable]  # value = <GameVariable.USER30: 101>
    USER31: typing.ClassVar[GameVariable]  # value = <GameVariable.USER31: 102>
    USER32: typing.ClassVar[GameVariable]  # value = <GameVariable.USER32: 103>
    USER33: typing.ClassVar[GameVariable]  # value = <GameVariable.USER33: 104>
    USER34: typing.ClassVar[GameVariable]  # value = <GameVariable.USER34: 105>
    USER35: typing.ClassVar[GameVariable]  # value = <GameVariable.USER35: 106>
    USER36: typing.ClassVar[GameVariable]  # value = <GameVariable.USER36: 107>
    USER37: typing.ClassVar[GameVariable]  # value = <GameVariable.USER37: 108>
    USER38: typing.ClassVar[GameVariable]  # value = <GameVariable.USER38: 109>
    USER39: typing.ClassVar[GameVariable]  # value = <GameVariable.USER39: 110>
    USER4: typing.ClassVar[GameVariable]  # value = <GameVariable.USER4: 75>
    USER40: typing.ClassVar[GameVariable]  # value = <GameVariable.USER40: 111>
    USER41: typing.ClassVar[GameVariable]  # value = <GameVariable.USER41: 112>
    USER42: typing.ClassVar[GameVariable]  # value = <GameVariable.USER42: 113>
    USER43: typing.ClassVar[GameVariable]  # value = <GameVariable.USER43: 114>
    USER44: typing.ClassVar[GameVariable]  # value = <GameVariable.USER44: 115>
    USER45: typing.ClassVar[GameVariable]  # value = <GameVariable.USER45: 116>
    USER46: typing.ClassVar[GameVariable]  # value = <GameVariable.USER46: 117>
    USER47: typing.ClassVar[GameVariable]  # value = <GameVariable.USER47: 118>
    USER48: typing.ClassVar[GameVariable]  # value = <GameVariable.USER48: 119>
    USER49: typing.ClassVar[GameVariable]  # value = <GameVariable.USER49: 120>
    USER5: typing.ClassVar[GameVariable]  # value = <GameVariable.USER5: 76>
    USER50: typing.ClassVar[GameVariable]  # value = <GameVariable.USER50: 121>
    USER51: typing.ClassVar[GameVariable]  # value = <GameVariable.USER51: 122>
    USER52: typing.ClassVar[GameVariable]  # value = <GameVariable.USER52: 123>
    USER53: typing.ClassVar[GameVariable]  # value = <GameVariable.USER53: 124>
    USER54: typing.ClassVar[GameVariable]  # value = <GameVariable.USER54: 125>
    USER55: typing.ClassVar[GameVariable]  # value = <GameVariable.USER55: 126>
    USER56: typing.ClassVar[GameVariable]  # value = <GameVariable.USER56: 127>
    USER57: typing.ClassVar[GameVariable]  # value = <GameVariable.USER57: 128>
    USER58: typing.ClassVar[GameVariable]  # value = <GameVariable.USER58: 129>
    USER59: typing.ClassVar[GameVariable]  # value = <GameVariable.USER59: 130>
    USER6: typing.ClassVar[GameVariable]  # value = <GameVariable.USER6: 77>
    USER60: typing.ClassVar[GameVariable]  # value = <GameVariable.USER60: 131>
    USER7: typing.ClassVar[GameVariable]  # value = <GameVariable.USER7: 78>
    USER8: typing.ClassVar[GameVariable]  # value = <GameVariable.USER8: 79>
    USER9: typing.ClassVar[GameVariable]  # value = <GameVariable.USER9: 80>
    VELOCITY_X: typing.ClassVar[GameVariable]  # value = <GameVariable.VELOCITY_X: 44>
    VELOCITY_Y: typing.ClassVar[GameVariable]  # value = <GameVariable.VELOCITY_Y: 45>
    VELOCITY_Z: typing.ClassVar[GameVariable]  # value = <GameVariable.VELOCITY_Z: 46>
    VIEW_HEIGHT: typing.ClassVar[GameVariable]  # value = <GameVariable.VIEW_HEIGHT: 43>
    WEAPON0: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON0: 27>
    WEAPON1: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON1: 28>
    WEAPON2: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON2: 29>
    WEAPON3: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON3: 30>
    WEAPON4: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON4: 31>
    WEAPON5: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON5: 32>
    WEAPON6: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON6: 33>
    WEAPON7: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON7: 34>
    WEAPON8: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON8: 35>
    WEAPON9: typing.ClassVar[GameVariable]  # value = <GameVariable.WEAPON9: 36>
    __members__: typing.ClassVar[
        dict[str, GameVariable]
    ]  # value = {'KILLCOUNT': <GameVariable.KILLCOUNT: 0>, 'ITEMCOUNT': <GameVariable.ITEMCOUNT: 1>, 'SECRETCOUNT': <GameVariable.SECRETCOUNT: 2>, 'FRAGCOUNT': <GameVariable.FRAGCOUNT: 3>, 'DEATHCOUNT': <GameVariable.DEATHCOUNT: 4>, 'HITCOUNT': <GameVariable.HITCOUNT: 5>, 'HITS_TAKEN': <GameVariable.HITS_TAKEN: 6>, 'DAMAGECOUNT': <GameVariable.DAMAGECOUNT: 7>, 'DAMAGE_TAKEN': <GameVariable.DAMAGE_TAKEN: 8>, 'HEALTH': <GameVariable.HEALTH: 9>, 'ARMOR': <GameVariable.ARMOR: 10>, 'DEAD': <GameVariable.DEAD: 11>, 'ON_GROUND': <GameVariable.ON_GROUND: 12>, 'ATTACK_READY': <GameVariable.ATTACK_READY: 13>, 'ALTATTACK_READY': <GameVariable.ALTATTACK_READY: 14>, 'SELECTED_WEAPON': <GameVariable.SELECTED_WEAPON: 15>, 'SELECTED_WEAPON_AMMO': <GameVariable.SELECTED_WEAPON_AMMO: 16>, 'AMMO1': <GameVariable.AMMO1: 18>, 'AMMO2': <GameVariable.AMMO2: 19>, 'AMMO3': <GameVariable.AMMO3: 20>, 'AMMO4': <GameVariable.AMMO4: 21>, 'AMMO5': <GameVariable.AMMO5: 22>, 'AMMO6': <GameVariable.AMMO6: 23>, 'AMMO7': <GameVariable.AMMO7: 24>, 'AMMO8': <GameVariable.AMMO8: 25>, 'AMMO9': <GameVariable.AMMO9: 26>, 'AMMO0': <GameVariable.AMMO0: 17>, 'WEAPON1': <GameVariable.WEAPON1: 28>, 'WEAPON2': <GameVariable.WEAPON2: 29>, 'WEAPON3': <GameVariable.WEAPON3: 30>, 'WEAPON4': <GameVariable.WEAPON4: 31>, 'WEAPON5': <GameVariable.WEAPON5: 32>, 'WEAPON6': <GameVariable.WEAPON6: 33>, 'WEAPON7': <GameVariable.WEAPON7: 34>, 'WEAPON8': <GameVariable.WEAPON8: 35>, 'WEAPON9': <GameVariable.WEAPON9: 36>, 'WEAPON0': <GameVariable.WEAPON0: 27>, 'POSITION_X': <GameVariable.POSITION_X: 37>, 'POSITION_Y': <GameVariable.POSITION_Y: 38>, 'POSITION_Z': <GameVariable.POSITION_Z: 39>, 'ANGLE': <GameVariable.ANGLE: 40>, 'PITCH': <GameVariable.PITCH: 41>, 'ROLL': <GameVariable.ROLL: 42>, 'VIEW_HEIGHT': <GameVariable.VIEW_HEIGHT: 43>, 'VELOCITY_X': <GameVariable.VELOCITY_X: 44>, 'VELOCITY_Y': <GameVariable.VELOCITY_Y: 45>, 'VELOCITY_Z': <GameVariable.VELOCITY_Z: 46>, 'CAMERA_POSITION_X': <GameVariable.CAMERA_POSITION_X: 47>, 'CAMERA_POSITION_Y': <GameVariable.CAMERA_POSITION_Y: 48>, 'CAMERA_POSITION_Z': <GameVariable.CAMERA_POSITION_Z: 49>, 'CAMERA_ANGLE': <GameVariable.CAMERA_ANGLE: 50>, 'CAMERA_PITCH': <GameVariable.CAMERA_PITCH: 51>, 'CAMERA_ROLL': <GameVariable.CAMERA_ROLL: 52>, 'CAMERA_FOV': <GameVariable.CAMERA_FOV: 53>, 'USER1': <GameVariable.USER1: 72>, 'USER2': <GameVariable.USER2: 73>, 'USER3': <GameVariable.USER3: 74>, 'USER4': <GameVariable.USER4: 75>, 'USER5': <GameVariable.USER5: 76>, 'USER6': <GameVariable.USER6: 77>, 'USER7': <GameVariable.USER7: 78>, 'USER8': <GameVariable.USER8: 79>, 'USER9': <GameVariable.USER9: 80>, 'USER10': <GameVariable.USER10: 81>, 'USER11': <GameVariable.USER11: 82>, 'USER12': <GameVariable.USER12: 83>, 'USER13': <GameVariable.USER13: 84>, 'USER14': <GameVariable.USER14: 85>, 'USER15': <GameVariable.USER15: 86>, 'USER16': <GameVariable.USER16: 87>, 'USER17': <GameVariable.USER17: 88>, 'USER18': <GameVariable.USER18: 89>, 'USER19': <GameVariable.USER19: 90>, 'USER20': <GameVariable.USER20: 91>, 'USER21': <GameVariable.USER21: 92>, 'USER22': <GameVariable.USER22: 93>, 'USER23': <GameVariable.USER23: 94>, 'USER24': <GameVariable.USER24: 95>, 'USER25': <GameVariable.USER25: 96>, 'USER26': <GameVariable.USER26: 97>, 'USER27': <GameVariable.USER27: 98>, 'USER28': <GameVariable.USER28: 99>, 'USER29': <GameVariable.USER29: 100>, 'USER30': <GameVariable.USER30: 101>, 'USER31': <GameVariable.USER31: 102>, 'USER32': <GameVariable.USER32: 103>, 'USER33': <GameVariable.USER33: 104>, 'USER34': <GameVariable.USER34: 105>, 'USER35': <GameVariable.USER35: 106>, 'USER36': <GameVariable.USER36: 107>, 'USER37': <GameVariable.USER37: 108>, 'USER38': <GameVariable.USER38: 109>, 'USER39': <GameVariable.USER39: 110>, 'USER40': <GameVariable.USER40: 111>, 'USER41': <GameVariable.USER41: 112>, 'USER42': <GameVariable.USER42: 113>, 'USER43': <GameVariable.USER43: 114>, 'USER44': <GameVariable.USER44: 115>, 'USER45': <GameVariable.USER45: 116>, 'USER46': <GameVariable.USER46: 117>, 'USER47': <GameVariable.USER47: 118>, 'USER48': <GameVariable.USER48: 119>, 'USER49': <GameVariable.USER49: 120>, 'USER50': <GameVariable.USER50: 121>, 'USER51': <GameVariable.USER51: 122>, 'USER52': <GameVariable.USER52: 123>, 'USER53': <GameVariable.USER53: 124>, 'USER54': <GameVariable.USER54: 125>, 'USER55': <GameVariable.USER55: 126>, 'USER56': <GameVariable.USER56: 127>, 'USER57': <GameVariable.USER57: 128>, 'USER58': <GameVariable.USER58: 129>, 'USER59': <GameVariable.USER59: 130>, 'USER60': <GameVariable.USER60: 131>, 'PLAYER_NUMBER': <GameVariable.PLAYER_NUMBER: 54>, 'PLAYER_COUNT': <GameVariable.PLAYER_COUNT: 55>, 'PLAYER1_FRAGCOUNT': <GameVariable.PLAYER1_FRAGCOUNT: 56>, 'PLAYER2_FRAGCOUNT': <GameVariable.PLAYER2_FRAGCOUNT: 57>, 'PLAYER3_FRAGCOUNT': <GameVariable.PLAYER3_FRAGCOUNT: 58>, 'PLAYER4_FRAGCOUNT': <GameVariable.PLAYER4_FRAGCOUNT: 59>, 'PLAYER5_FRAGCOUNT': <GameVariable.PLAYER5_FRAGCOUNT: 60>, 'PLAYER6_FRAGCOUNT': <GameVariable.PLAYER6_FRAGCOUNT: 61>, 'PLAYER7_FRAGCOUNT': <GameVariable.PLAYER7_FRAGCOUNT: 62>, 'PLAYER8_FRAGCOUNT': <GameVariable.PLAYER8_FRAGCOUNT: 63>, 'PLAYER9_FRAGCOUNT': <GameVariable.PLAYER9_FRAGCOUNT: 64>, 'PLAYER10_FRAGCOUNT': <GameVariable.PLAYER10_FRAGCOUNT: 65>, 'PLAYER11_FRAGCOUNT': <GameVariable.PLAYER11_FRAGCOUNT: 66>, 'PLAYER12_FRAGCOUNT': <GameVariable.PLAYER12_FRAGCOUNT: 67>, 'PLAYER13_FRAGCOUNT': <GameVariable.PLAYER13_FRAGCOUNT: 68>, 'PLAYER14_FRAGCOUNT': <GameVariable.PLAYER14_FRAGCOUNT: 69>, 'PLAYER15_FRAGCOUNT': <GameVariable.PLAYER15_FRAGCOUNT: 70>, 'PLAYER16_FRAGCOUNT': <GameVariable.PLAYER16_FRAGCOUNT: 71>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Label:
    """
    Represents object labels in the game world with associated properties.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def height(self) -> int: ...
    @property
    def object_angle(self) -> float: ...
    @property
    def object_category(self) -> str: ...
    @property
    def object_id(self) -> int: ...
    @property
    def object_name(self) -> str: ...
    @property
    def object_pitch(self) -> float: ...
    @property
    def object_position_x(self) -> float: ...
    @property
    def object_position_y(self) -> float: ...
    @property
    def object_position_z(self) -> float: ...
    @property
    def object_roll(self) -> float: ...
    @property
    def object_velocity_x(self) -> float: ...
    @property
    def object_velocity_y(self) -> float: ...
    @property
    def object_velocity_z(self) -> float: ...
    @property
    def value(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def x(self) -> int: ...
    @property
    def y(self) -> int: ...

class Line:
    """
    Represents line segments in the game world geometry.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def is_blocking(self) -> bool: ...
    @property
    def x1(self) -> float: ...
    @property
    def x2(self) -> float: ...
    @property
    def y1(self) -> float: ...
    @property
    def y2(self) -> float: ...

class MessageQueueException(Exception):
    pass

class Mode:
    """
    Defines the mode for controlling the game.

    Members:

      PLAYER

      SPECTATOR

      ASYNC_PLAYER

      ASYNC_SPECTATOR
    """

    ASYNC_PLAYER: typing.ClassVar[Mode]  # value = <Mode.ASYNC_PLAYER: 2>
    ASYNC_SPECTATOR: typing.ClassVar[Mode]  # value = <Mode.ASYNC_SPECTATOR: 3>
    PLAYER: typing.ClassVar[Mode]  # value = <Mode.PLAYER: 0>
    SPECTATOR: typing.ClassVar[Mode]  # value = <Mode.SPECTATOR: 1>
    __members__: typing.ClassVar[
        dict[str, Mode]
    ]  # value = {'PLAYER': <Mode.PLAYER: 0>, 'SPECTATOR': <Mode.SPECTATOR: 1>, 'ASYNC_PLAYER': <Mode.ASYNC_PLAYER: 2>, 'ASYNC_SPECTATOR': <Mode.ASYNC_SPECTATOR: 3>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Object:
    """
    Represents objects in the game world with position and other properties.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def angle(self) -> float: ...
    @property
    def id(self) -> int: ...
    @property
    def name(self) -> str: ...
    @property
    def pitch(self) -> float: ...
    @property
    def position_x(self) -> float: ...
    @property
    def position_y(self) -> float: ...
    @property
    def position_z(self) -> float: ...
    @property
    def roll(self) -> float: ...
    @property
    def velocity_x(self) -> float: ...
    @property
    def velocity_y(self) -> float: ...
    @property
    def velocity_z(self) -> float: ...

class SamplingRate:
    """
    Defines available audio sampling rates.

    Members:

      SR_11025

      SR_22050

      SR_44100
    """

    SR_11025: typing.ClassVar[SamplingRate]  # value = <SamplingRate.SR_11025: 0>
    SR_22050: typing.ClassVar[SamplingRate]  # value = <SamplingRate.SR_22050: 1>
    SR_44100: typing.ClassVar[SamplingRate]  # value = <SamplingRate.SR_44100: 2>
    __members__: typing.ClassVar[
        dict[str, SamplingRate]
    ]  # value = {'SR_11025': <SamplingRate.SR_11025: 0>, 'SR_22050': <SamplingRate.SR_22050: 1>, 'SR_44100': <SamplingRate.SR_44100: 2>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ScreenFormat:
    """
    Defines the format of the screen buffer.

    Members:

      CRCGCB

      RGB24

      RGBA32

      ARGB32

      CBCGCR

      BGR24

      BGRA32

      ABGR32

      GRAY8

      DOOM_256_COLORS8
    """

    ABGR32: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.ABGR32: 7>
    ARGB32: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.ARGB32: 3>
    BGR24: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.BGR24: 5>
    BGRA32: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.BGRA32: 6>
    CBCGCR: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.CBCGCR: 4>
    CRCGCB: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.CRCGCB: 0>
    DOOM_256_COLORS8: typing.ClassVar[
        ScreenFormat
    ]  # value = <ScreenFormat.DOOM_256_COLORS8: 9>
    GRAY8: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.GRAY8: 8>
    RGB24: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.RGB24: 1>
    RGBA32: typing.ClassVar[ScreenFormat]  # value = <ScreenFormat.RGBA32: 2>
    __members__: typing.ClassVar[
        dict[str, ScreenFormat]
    ]  # value = {'CRCGCB': <ScreenFormat.CRCGCB: 0>, 'RGB24': <ScreenFormat.RGB24: 1>, 'RGBA32': <ScreenFormat.RGBA32: 2>, 'ARGB32': <ScreenFormat.ARGB32: 3>, 'CBCGCR': <ScreenFormat.CBCGCR: 4>, 'BGR24': <ScreenFormat.BGR24: 5>, 'BGRA32': <ScreenFormat.BGRA32: 6>, 'ABGR32': <ScreenFormat.ABGR32: 7>, 'GRAY8': <ScreenFormat.GRAY8: 8>, 'DOOM_256_COLORS8': <ScreenFormat.DOOM_256_COLORS8: 9>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class ScreenResolution:
    """
    Defines the resolution of the screen buffer. Available resolutions include various predefined sizes like RES_320x240, etc.

    Members:

      RES_160X120

      RES_200X125

      RES_200X150

      RES_256X144

      RES_256X160

      RES_256X192

      RES_320X180

      RES_320X200

      RES_320X240

      RES_320X256

      RES_400X225

      RES_400X250

      RES_400X300

      RES_512X288

      RES_512X320

      RES_512X384

      RES_640X360

      RES_640X400

      RES_640X480

      RES_800X450

      RES_800X500

      RES_800X600

      RES_1024X576

      RES_1024X640

      RES_1024X768

      RES_1280X720

      RES_1280X800

      RES_1280X960

      RES_1280X1024

      RES_1400X787

      RES_1400X875

      RES_1400X1050

      RES_1600X900

      RES_1600X1000

      RES_1600X1200

      RES_1920X1080
    """

    RES_1024X576: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1024X576: 22>
    RES_1024X640: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1024X640: 23>
    RES_1024X768: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1024X768: 24>
    RES_1280X1024: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1280X1024: 28>
    RES_1280X720: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1280X720: 25>
    RES_1280X800: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1280X800: 26>
    RES_1280X960: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1280X960: 27>
    RES_1400X1050: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1400X1050: 31>
    RES_1400X787: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1400X787: 29>
    RES_1400X875: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1400X875: 30>
    RES_1600X1000: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1600X1000: 33>
    RES_1600X1200: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1600X1200: 34>
    RES_1600X900: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1600X900: 32>
    RES_160X120: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_160X120: 0>
    RES_1920X1080: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_1920X1080: 35>
    RES_200X125: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_200X125: 1>
    RES_200X150: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_200X150: 2>
    RES_256X144: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_256X144: 3>
    RES_256X160: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_256X160: 4>
    RES_256X192: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_256X192: 5>
    RES_320X180: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_320X180: 6>
    RES_320X200: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_320X200: 7>
    RES_320X240: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_320X240: 8>
    RES_320X256: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_320X256: 9>
    RES_400X225: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_400X225: 10>
    RES_400X250: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_400X250: 11>
    RES_400X300: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_400X300: 12>
    RES_512X288: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_512X288: 13>
    RES_512X320: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_512X320: 14>
    RES_512X384: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_512X384: 15>
    RES_640X360: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_640X360: 16>
    RES_640X400: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_640X400: 17>
    RES_640X480: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_640X480: 18>
    RES_800X450: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_800X450: 19>
    RES_800X500: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_800X500: 20>
    RES_800X600: typing.ClassVar[
        ScreenResolution
    ]  # value = <ScreenResolution.RES_800X600: 21>
    __members__: typing.ClassVar[
        dict[str, ScreenResolution]
    ]  # value = {'RES_160X120': <ScreenResolution.RES_160X120: 0>, 'RES_200X125': <ScreenResolution.RES_200X125: 1>, 'RES_200X150': <ScreenResolution.RES_200X150: 2>, 'RES_256X144': <ScreenResolution.RES_256X144: 3>, 'RES_256X160': <ScreenResolution.RES_256X160: 4>, 'RES_256X192': <ScreenResolution.RES_256X192: 5>, 'RES_320X180': <ScreenResolution.RES_320X180: 6>, 'RES_320X200': <ScreenResolution.RES_320X200: 7>, 'RES_320X240': <ScreenResolution.RES_320X240: 8>, 'RES_320X256': <ScreenResolution.RES_320X256: 9>, 'RES_400X225': <ScreenResolution.RES_400X225: 10>, 'RES_400X250': <ScreenResolution.RES_400X250: 11>, 'RES_400X300': <ScreenResolution.RES_400X300: 12>, 'RES_512X288': <ScreenResolution.RES_512X288: 13>, 'RES_512X320': <ScreenResolution.RES_512X320: 14>, 'RES_512X384': <ScreenResolution.RES_512X384: 15>, 'RES_640X360': <ScreenResolution.RES_640X360: 16>, 'RES_640X400': <ScreenResolution.RES_640X400: 17>, 'RES_640X480': <ScreenResolution.RES_640X480: 18>, 'RES_800X450': <ScreenResolution.RES_800X450: 19>, 'RES_800X500': <ScreenResolution.RES_800X500: 20>, 'RES_800X600': <ScreenResolution.RES_800X600: 21>, 'RES_1024X576': <ScreenResolution.RES_1024X576: 22>, 'RES_1024X640': <ScreenResolution.RES_1024X640: 23>, 'RES_1024X768': <ScreenResolution.RES_1024X768: 24>, 'RES_1280X720': <ScreenResolution.RES_1280X720: 25>, 'RES_1280X800': <ScreenResolution.RES_1280X800: 26>, 'RES_1280X960': <ScreenResolution.RES_1280X960: 27>, 'RES_1280X1024': <ScreenResolution.RES_1280X1024: 28>, 'RES_1400X787': <ScreenResolution.RES_1400X787: 29>, 'RES_1400X875': <ScreenResolution.RES_1400X875: 30>, 'RES_1400X1050': <ScreenResolution.RES_1400X1050: 31>, 'RES_1600X900': <ScreenResolution.RES_1600X900: 32>, 'RES_1600X1000': <ScreenResolution.RES_1600X1000: 33>, 'RES_1600X1200': <ScreenResolution.RES_1600X1200: 34>, 'RES_1920X1080': <ScreenResolution.RES_1920X1080: 35>}
    def __eq__(self, other: typing.Any) -> bool: ...
    def __getstate__(self) -> int: ...
    def __hash__(self) -> int: ...
    def __index__(self) -> int: ...
    def __init__(self, value: int) -> None: ...
    def __int__(self) -> int: ...
    def __ne__(self, other: typing.Any) -> bool: ...
    def __repr__(self) -> str: ...
    def __setstate__(self, state: int) -> None: ...
    def __str__(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def value(self) -> int: ...

class Sector:
    """
    Represents sectors (floor/ceiling areas) in the game world geometry.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def ceiling_height(self) -> float: ...
    @property
    def floor_height(self) -> float: ...
    @property
    def lines(self) -> list: ...

class ServerState:
    """
    Contains the state of the multiplayer server.
    """

    def __getstate__(self) -> tuple: ...
    def __setstate__(self, arg0: tuple) -> None: ...
    @property
    def player_count(self) -> int: ...
    @property
    def players_afk(self) -> list: ...
    @property
    def players_frags(self) -> list: ...
    @property
    def players_in_game(self) -> list: ...
    @property
    def players_last_action_tic(self) -> list: ...
    @property
    def players_last_kill_tic(self) -> list: ...
    @property
    def players_names(self) -> list: ...
    @property
    def tic(self) -> int: ...

class SharedMemoryException(Exception):
    pass

class ViZDoomErrorException(Exception):
    pass

class ViZDoomIsNotRunningException(Exception):
    pass

class ViZDoomNoOpenALSoundException(Exception):
    pass

class ViZDoomUnexpectedExitException(Exception):
    pass

@typing.overload
def doom_fixed_to_double(doom_fixed: int) -> float:
    """
    Converts fixed point numeral to a floating point value.
    Doom engine internally use fixed point numbers.
    If you assign fixed point numeral to ``USER1`` - ``USER60`` GameVariables,
    you can convert them to floating point by using this function.
    """

@typing.overload
def doom_fixed_to_double(doom_fixed: float) -> float:
    """
    Converts fixed point numeral to a floating point value.
    Doom engine internally use fixed point numbers.
    If you assign fixed point numeral to ``USER1`` - ``USER60`` GameVariables,
    you can convert them to floating point by using this function.
    """

@typing.overload
def doom_fixed_to_float(doom_fixed: int) -> float:
    """
    Converts fixed point numeral to a floating point value.
    Doom engine internally use fixed point numbers.
    If you assign fixed point numeral to ``USER1`` - ``USER60`` GameVariables,
    you can convert them to floating point by using this function.
    """

@typing.overload
def doom_fixed_to_float(doom_fixed: float) -> float:
    """
    Converts fixed point numeral to a floating point value.
    Doom engine internally use fixed point numbers.
    If you assign fixed point numeral to ``USER1`` - ``USER60`` GameVariables,
    you can convert them to floating point by using this function.
    """

def doom_tics_to_ms(doom_tics: float, fps: int = 35) -> float:
    """
    Calculates how many tics will be made during given number of milliseconds.

    Note: changed in 1.1.0
    """

def doom_tics_to_sec(doom_tics: float, fps: int = 35) -> float:
    """
    Calculates how many tics will be made during given number of seconds.

    Note: added in 1.1.0
    """

def get_default_categories() -> list[str]:
    """
    Returns the default object categories of ViZDoom.

    Note: added in 1.3.0.
    """

def is_binary_button(button: Button) -> bool:
    """
    Returns ``True`` if :class:`.Button` is binary button.
    """

def is_delta_button(button: Button) -> bool:
    """
    Returns ``True`` if :class:`.Button` is delta button.
    """

def ms_to_doom_tics(doom_tics: float, fps: int = 35) -> float:
    """
    Calculates the number of milliseconds that will pass during specified number of tics.

    Note: changed in 1.1.0
    """

def sec_to_doom_tics(doom_tics: float, fps: int = 35) -> float:
    """
    Calculates the number of seconds that will pass during specified number of tics.

    Note: added in 1.1.0
    """

ABGR32: ScreenFormat  # value = <ScreenFormat.ABGR32: 7>
ACTIVATE_SELECTED_ITEM: Button  # value = <Button.ACTIVATE_SELECTED_ITEM: 34>
ALTATTACK: Button  # value = <Button.ALTATTACK: 5>
ALTATTACK_READY: GameVariable  # value = <GameVariable.ALTATTACK_READY: 14>
AMMO0: GameVariable  # value = <GameVariable.AMMO0: 17>
AMMO1: GameVariable  # value = <GameVariable.AMMO1: 18>
AMMO2: GameVariable  # value = <GameVariable.AMMO2: 19>
AMMO3: GameVariable  # value = <GameVariable.AMMO3: 20>
AMMO4: GameVariable  # value = <GameVariable.AMMO4: 21>
AMMO5: GameVariable  # value = <GameVariable.AMMO5: 22>
AMMO6: GameVariable  # value = <GameVariable.AMMO6: 23>
AMMO7: GameVariable  # value = <GameVariable.AMMO7: 24>
AMMO8: GameVariable  # value = <GameVariable.AMMO8: 25>
AMMO9: GameVariable  # value = <GameVariable.AMMO9: 26>
ANGLE: GameVariable  # value = <GameVariable.ANGLE: 40>
ARGB32: ScreenFormat  # value = <ScreenFormat.ARGB32: 3>
ARMOR: GameVariable  # value = <GameVariable.ARMOR: 10>
ASYNC_PLAYER: Mode  # value = <Mode.ASYNC_PLAYER: 2>
ASYNC_SPECTATOR: Mode  # value = <Mode.ASYNC_SPECTATOR: 3>
ATTACK: Button  # value = <Button.ATTACK: 0>
ATTACK_READY: GameVariable  # value = <GameVariable.ATTACK_READY: 13>
BGR24: ScreenFormat  # value = <ScreenFormat.BGR24: 5>
BGRA32: ScreenFormat  # value = <ScreenFormat.BGRA32: 6>
BINARY_BUTTON_COUNT: int = 38
BUTTON_COUNT: int = 43
CAMERA_ANGLE: GameVariable  # value = <GameVariable.CAMERA_ANGLE: 50>
CAMERA_FOV: GameVariable  # value = <GameVariable.CAMERA_FOV: 53>
CAMERA_PITCH: GameVariable  # value = <GameVariable.CAMERA_PITCH: 51>
CAMERA_POSITION_X: GameVariable  # value = <GameVariable.CAMERA_POSITION_X: 47>
CAMERA_POSITION_Y: GameVariable  # value = <GameVariable.CAMERA_POSITION_Y: 48>
CAMERA_POSITION_Z: GameVariable  # value = <GameVariable.CAMERA_POSITION_Z: 49>
CAMERA_ROLL: GameVariable  # value = <GameVariable.CAMERA_ROLL: 52>
CBCGCR: ScreenFormat  # value = <ScreenFormat.CBCGCR: 4>
CRCGCB: ScreenFormat  # value = <ScreenFormat.CRCGCB: 0>
CROUCH: Button  # value = <Button.CROUCH: 3>
DAMAGECOUNT: GameVariable  # value = <GameVariable.DAMAGECOUNT: 7>
DAMAGE_TAKEN: GameVariable  # value = <GameVariable.DAMAGE_TAKEN: 8>
DEAD: GameVariable  # value = <GameVariable.DEAD: 11>
DEATHCOUNT: GameVariable  # value = <GameVariable.DEATHCOUNT: 4>
DEFAULT_FPS: int = 35
DEFAULT_FRAMETIME_MS: float = 28.57142857142857
DEFAULT_FRAMETIME_S: float = 0.02857142857142857
DEFAULT_TICRATE: int = 35
DELTA_BUTTON_COUNT: int = 5
DOOM_256_COLORS8: ScreenFormat  # value = <ScreenFormat.DOOM_256_COLORS8: 9>
DROP_SELECTED_ITEM: Button  # value = <Button.DROP_SELECTED_ITEM: 37>
DROP_SELECTED_WEAPON: Button  # value = <Button.DROP_SELECTED_WEAPON: 33>
FRAGCOUNT: GameVariable  # value = <GameVariable.FRAGCOUNT: 3>
GRAY8: ScreenFormat  # value = <ScreenFormat.GRAY8: 8>
HEALTH: GameVariable  # value = <GameVariable.HEALTH: 9>
HITCOUNT: GameVariable  # value = <GameVariable.HITCOUNT: 5>
HITS_TAKEN: GameVariable  # value = <GameVariable.HITS_TAKEN: 6>
ITEMCOUNT: GameVariable  # value = <GameVariable.ITEMCOUNT: 1>
JUMP: Button  # value = <Button.JUMP: 2>
KILLCOUNT: GameVariable  # value = <GameVariable.KILLCOUNT: 0>
LAND: Button  # value = <Button.LAND: 20>
LOOK_DOWN: Button  # value = <Button.LOOK_DOWN: 17>
LOOK_UP: Button  # value = <Button.LOOK_UP: 16>
LOOK_UP_DOWN_DELTA: Button  # value = <Button.LOOK_UP_DOWN_DELTA: 38>
MAX_PLAYERS: int = 16
MAX_PLAYER_NAME_LENGTH: int = 128
MOVE_BACKWARD: Button  # value = <Button.MOVE_BACKWARD: 12>
MOVE_DOWN: Button  # value = <Button.MOVE_DOWN: 19>
MOVE_FORWARD: Button  # value = <Button.MOVE_FORWARD: 13>
MOVE_FORWARD_BACKWARD_DELTA: Button  # value = <Button.MOVE_FORWARD_BACKWARD_DELTA: 40>
MOVE_LEFT: Button  # value = <Button.MOVE_LEFT: 11>
MOVE_LEFT_RIGHT_DELTA: Button  # value = <Button.MOVE_LEFT_RIGHT_DELTA: 41>
MOVE_RIGHT: Button  # value = <Button.MOVE_RIGHT: 10>
MOVE_UP: Button  # value = <Button.MOVE_UP: 18>
MOVE_UP_DOWN_DELTA: Button  # value = <Button.MOVE_UP_DOWN_DELTA: 42>
NORMAL: AutomapMode  # value = <AutomapMode.NORMAL: 0>
OBJECTS: AutomapMode  # value = <AutomapMode.OBJECTS: 2>
OBJECTS_WITH_SIZE: AutomapMode  # value = <AutomapMode.OBJECTS_WITH_SIZE: 3>
ON_GROUND: GameVariable  # value = <GameVariable.ON_GROUND: 12>
PITCH: GameVariable  # value = <GameVariable.PITCH: 41>
PLAYER: Mode  # value = <Mode.PLAYER: 0>
PLAYER10_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER10_FRAGCOUNT: 65>
PLAYER11_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER11_FRAGCOUNT: 66>
PLAYER12_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER12_FRAGCOUNT: 67>
PLAYER13_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER13_FRAGCOUNT: 68>
PLAYER14_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER14_FRAGCOUNT: 69>
PLAYER15_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER15_FRAGCOUNT: 70>
PLAYER16_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER16_FRAGCOUNT: 71>
PLAYER1_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER1_FRAGCOUNT: 56>
PLAYER2_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER2_FRAGCOUNT: 57>
PLAYER3_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER3_FRAGCOUNT: 58>
PLAYER4_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER4_FRAGCOUNT: 59>
PLAYER5_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER5_FRAGCOUNT: 60>
PLAYER6_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER6_FRAGCOUNT: 61>
PLAYER7_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER7_FRAGCOUNT: 62>
PLAYER8_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER8_FRAGCOUNT: 63>
PLAYER9_FRAGCOUNT: GameVariable  # value = <GameVariable.PLAYER9_FRAGCOUNT: 64>
PLAYER_COUNT: GameVariable  # value = <GameVariable.PLAYER_COUNT: 55>
PLAYER_NUMBER: GameVariable  # value = <GameVariable.PLAYER_NUMBER: 54>
POSITION_X: GameVariable  # value = <GameVariable.POSITION_X: 37>
POSITION_Y: GameVariable  # value = <GameVariable.POSITION_Y: 38>
POSITION_Z: GameVariable  # value = <GameVariable.POSITION_Z: 39>
RELOAD: Button  # value = <Button.RELOAD: 6>
RES_1024X576: ScreenResolution  # value = <ScreenResolution.RES_1024X576: 22>
RES_1024X640: ScreenResolution  # value = <ScreenResolution.RES_1024X640: 23>
RES_1024X768: ScreenResolution  # value = <ScreenResolution.RES_1024X768: 24>
RES_1280X1024: ScreenResolution  # value = <ScreenResolution.RES_1280X1024: 28>
RES_1280X720: ScreenResolution  # value = <ScreenResolution.RES_1280X720: 25>
RES_1280X800: ScreenResolution  # value = <ScreenResolution.RES_1280X800: 26>
RES_1280X960: ScreenResolution  # value = <ScreenResolution.RES_1280X960: 27>
RES_1400X1050: ScreenResolution  # value = <ScreenResolution.RES_1400X1050: 31>
RES_1400X787: ScreenResolution  # value = <ScreenResolution.RES_1400X787: 29>
RES_1400X875: ScreenResolution  # value = <ScreenResolution.RES_1400X875: 30>
RES_1600X1000: ScreenResolution  # value = <ScreenResolution.RES_1600X1000: 33>
RES_1600X1200: ScreenResolution  # value = <ScreenResolution.RES_1600X1200: 34>
RES_1600X900: ScreenResolution  # value = <ScreenResolution.RES_1600X900: 32>
RES_160X120: ScreenResolution  # value = <ScreenResolution.RES_160X120: 0>
RES_1920X1080: ScreenResolution  # value = <ScreenResolution.RES_1920X1080: 35>
RES_200X125: ScreenResolution  # value = <ScreenResolution.RES_200X125: 1>
RES_200X150: ScreenResolution  # value = <ScreenResolution.RES_200X150: 2>
RES_256X144: ScreenResolution  # value = <ScreenResolution.RES_256X144: 3>
RES_256X160: ScreenResolution  # value = <ScreenResolution.RES_256X160: 4>
RES_256X192: ScreenResolution  # value = <ScreenResolution.RES_256X192: 5>
RES_320X180: ScreenResolution  # value = <ScreenResolution.RES_320X180: 6>
RES_320X200: ScreenResolution  # value = <ScreenResolution.RES_320X200: 7>
RES_320X240: ScreenResolution  # value = <ScreenResolution.RES_320X240: 8>
RES_320X256: ScreenResolution  # value = <ScreenResolution.RES_320X256: 9>
RES_400X225: ScreenResolution  # value = <ScreenResolution.RES_400X225: 10>
RES_400X250: ScreenResolution  # value = <ScreenResolution.RES_400X250: 11>
RES_400X300: ScreenResolution  # value = <ScreenResolution.RES_400X300: 12>
RES_512X288: ScreenResolution  # value = <ScreenResolution.RES_512X288: 13>
RES_512X320: ScreenResolution  # value = <ScreenResolution.RES_512X320: 14>
RES_512X384: ScreenResolution  # value = <ScreenResolution.RES_512X384: 15>
RES_640X360: ScreenResolution  # value = <ScreenResolution.RES_640X360: 16>
RES_640X400: ScreenResolution  # value = <ScreenResolution.RES_640X400: 17>
RES_640X480: ScreenResolution  # value = <ScreenResolution.RES_640X480: 18>
RES_800X450: ScreenResolution  # value = <ScreenResolution.RES_800X450: 19>
RES_800X500: ScreenResolution  # value = <ScreenResolution.RES_800X500: 20>
RES_800X600: ScreenResolution  # value = <ScreenResolution.RES_800X600: 21>
RGB24: ScreenFormat  # value = <ScreenFormat.RGB24: 1>
RGBA32: ScreenFormat  # value = <ScreenFormat.RGBA32: 2>
ROLL: GameVariable  # value = <GameVariable.ROLL: 42>
SECRETCOUNT: GameVariable  # value = <GameVariable.SECRETCOUNT: 2>
SELECTED_WEAPON: GameVariable  # value = <GameVariable.SELECTED_WEAPON: 15>
SELECTED_WEAPON_AMMO: GameVariable  # value = <GameVariable.SELECTED_WEAPON_AMMO: 16>
SELECT_NEXT_ITEM: Button  # value = <Button.SELECT_NEXT_ITEM: 35>
SELECT_NEXT_WEAPON: Button  # value = <Button.SELECT_NEXT_WEAPON: 31>
SELECT_PREV_ITEM: Button  # value = <Button.SELECT_PREV_ITEM: 36>
SELECT_PREV_WEAPON: Button  # value = <Button.SELECT_PREV_WEAPON: 32>
SELECT_WEAPON0: Button  # value = <Button.SELECT_WEAPON0: 30>
SELECT_WEAPON1: Button  # value = <Button.SELECT_WEAPON1: 21>
SELECT_WEAPON2: Button  # value = <Button.SELECT_WEAPON2: 22>
SELECT_WEAPON3: Button  # value = <Button.SELECT_WEAPON3: 23>
SELECT_WEAPON4: Button  # value = <Button.SELECT_WEAPON4: 24>
SELECT_WEAPON5: Button  # value = <Button.SELECT_WEAPON5: 25>
SELECT_WEAPON6: Button  # value = <Button.SELECT_WEAPON6: 26>
SELECT_WEAPON7: Button  # value = <Button.SELECT_WEAPON7: 27>
SELECT_WEAPON8: Button  # value = <Button.SELECT_WEAPON8: 28>
SELECT_WEAPON9: Button  # value = <Button.SELECT_WEAPON9: 29>
SLOT_COUNT: int = 10
SPECTATOR: Mode  # value = <Mode.SPECTATOR: 1>
SPEED: Button  # value = <Button.SPEED: 8>
SR_11025: SamplingRate  # value = <SamplingRate.SR_11025: 0>
SR_22050: SamplingRate  # value = <SamplingRate.SR_22050: 1>
SR_44100: SamplingRate  # value = <SamplingRate.SR_44100: 2>
STRAFE: Button  # value = <Button.STRAFE: 9>
TURN180: Button  # value = <Button.TURN180: 4>
TURN_LEFT: Button  # value = <Button.TURN_LEFT: 15>
TURN_LEFT_RIGHT_DELTA: Button  # value = <Button.TURN_LEFT_RIGHT_DELTA: 39>
TURN_RIGHT: Button  # value = <Button.TURN_RIGHT: 14>
USE: Button  # value = <Button.USE: 1>
USER1: GameVariable  # value = <GameVariable.USER1: 72>
USER10: GameVariable  # value = <GameVariable.USER10: 81>
USER11: GameVariable  # value = <GameVariable.USER11: 82>
USER12: GameVariable  # value = <GameVariable.USER12: 83>
USER13: GameVariable  # value = <GameVariable.USER13: 84>
USER14: GameVariable  # value = <GameVariable.USER14: 85>
USER15: GameVariable  # value = <GameVariable.USER15: 86>
USER16: GameVariable  # value = <GameVariable.USER16: 87>
USER17: GameVariable  # value = <GameVariable.USER17: 88>
USER18: GameVariable  # value = <GameVariable.USER18: 89>
USER19: GameVariable  # value = <GameVariable.USER19: 90>
USER2: GameVariable  # value = <GameVariable.USER2: 73>
USER20: GameVariable  # value = <GameVariable.USER20: 91>
USER21: GameVariable  # value = <GameVariable.USER21: 92>
USER22: GameVariable  # value = <GameVariable.USER22: 93>
USER23: GameVariable  # value = <GameVariable.USER23: 94>
USER24: GameVariable  # value = <GameVariable.USER24: 95>
USER25: GameVariable  # value = <GameVariable.USER25: 96>
USER26: GameVariable  # value = <GameVariable.USER26: 97>
USER27: GameVariable  # value = <GameVariable.USER27: 98>
USER28: GameVariable  # value = <GameVariable.USER28: 99>
USER29: GameVariable  # value = <GameVariable.USER29: 100>
USER3: GameVariable  # value = <GameVariable.USER3: 74>
USER30: GameVariable  # value = <GameVariable.USER30: 101>
USER31: GameVariable  # value = <GameVariable.USER31: 102>
USER32: GameVariable  # value = <GameVariable.USER32: 103>
USER33: GameVariable  # value = <GameVariable.USER33: 104>
USER34: GameVariable  # value = <GameVariable.USER34: 105>
USER35: GameVariable  # value = <GameVariable.USER35: 106>
USER36: GameVariable  # value = <GameVariable.USER36: 107>
USER37: GameVariable  # value = <GameVariable.USER37: 108>
USER38: GameVariable  # value = <GameVariable.USER38: 109>
USER39: GameVariable  # value = <GameVariable.USER39: 110>
USER4: GameVariable  # value = <GameVariable.USER4: 75>
USER40: GameVariable  # value = <GameVariable.USER40: 111>
USER41: GameVariable  # value = <GameVariable.USER41: 112>
USER42: GameVariable  # value = <GameVariable.USER42: 113>
USER43: GameVariable  # value = <GameVariable.USER43: 114>
USER44: GameVariable  # value = <GameVariable.USER44: 115>
USER45: GameVariable  # value = <GameVariable.USER45: 116>
USER46: GameVariable  # value = <GameVariable.USER46: 117>
USER47: GameVariable  # value = <GameVariable.USER47: 118>
USER48: GameVariable  # value = <GameVariable.USER48: 119>
USER49: GameVariable  # value = <GameVariable.USER49: 120>
USER5: GameVariable  # value = <GameVariable.USER5: 76>
USER50: GameVariable  # value = <GameVariable.USER50: 121>
USER51: GameVariable  # value = <GameVariable.USER51: 122>
USER52: GameVariable  # value = <GameVariable.USER52: 123>
USER53: GameVariable  # value = <GameVariable.USER53: 124>
USER54: GameVariable  # value = <GameVariable.USER54: 125>
USER55: GameVariable  # value = <GameVariable.USER55: 126>
USER56: GameVariable  # value = <GameVariable.USER56: 127>
USER57: GameVariable  # value = <GameVariable.USER57: 128>
USER58: GameVariable  # value = <GameVariable.USER58: 129>
USER59: GameVariable  # value = <GameVariable.USER59: 130>
USER6: GameVariable  # value = <GameVariable.USER6: 77>
USER60: GameVariable  # value = <GameVariable.USER60: 131>
USER7: GameVariable  # value = <GameVariable.USER7: 78>
USER8: GameVariable  # value = <GameVariable.USER8: 79>
USER9: GameVariable  # value = <GameVariable.USER9: 80>
USER_VARIABLE_COUNT: int = 60
VELOCITY_X: GameVariable  # value = <GameVariable.VELOCITY_X: 44>
VELOCITY_Y: GameVariable  # value = <GameVariable.VELOCITY_Y: 45>
VELOCITY_Z: GameVariable  # value = <GameVariable.VELOCITY_Z: 46>
VIEW_HEIGHT: GameVariable  # value = <GameVariable.VIEW_HEIGHT: 43>
WEAPON0: GameVariable  # value = <GameVariable.WEAPON0: 27>
WEAPON1: GameVariable  # value = <GameVariable.WEAPON1: 28>
WEAPON2: GameVariable  # value = <GameVariable.WEAPON2: 29>
WEAPON3: GameVariable  # value = <GameVariable.WEAPON3: 30>
WEAPON4: GameVariable  # value = <GameVariable.WEAPON4: 31>
WEAPON5: GameVariable  # value = <GameVariable.WEAPON5: 32>
WEAPON6: GameVariable  # value = <GameVariable.WEAPON6: 33>
WEAPON7: GameVariable  # value = <GameVariable.WEAPON7: 34>
WEAPON8: GameVariable  # value = <GameVariable.WEAPON8: 35>
WEAPON9: GameVariable  # value = <GameVariable.WEAPON9: 36>
WHOLE: AutomapMode  # value = <AutomapMode.WHOLE: 1>
ZOOM: Button  # value = <Button.ZOOM: 7>
__version__: str
