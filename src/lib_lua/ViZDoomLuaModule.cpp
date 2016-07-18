/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.
*/

#include "ViZDoomDefines.h"
#include "ViZDoomExceptions.h"

#include "ViZDoomGameLua.h"

#include <iostream>
#include <string>
#include <vector>

#include <chrono>
#include <thread>

extern "C" {
    #include "lua.h"
    #include "lualib.h"
    #include "lauxlib.h"
}

#include <luabind/luabind.hpp>

using namespace vizdoom;
using namespace luabind;

void (DoomGameLua::*newEpisode1)() = &DoomGameLua::newEpisode;
void (DoomGameLua::*newEpisode2)(std::string) = &DoomGameLua::newEpisode;

void (DoomGameLua::*addAvailableButton1)(Button) = &DoomGameLua::addAvailableButton;
void (DoomGameLua::*addAvailableButton2)(Button, unsigned int) = &DoomGameLua::addAvailableButton;

void (DoomGameLua::*advanceAction1)() = &DoomGameLua::advanceAction;
void (DoomGameLua::*advanceAction2)(unsigned int) = &DoomGameLua::advanceAction;
void (DoomGameLua::*advanceAction3)(unsigned int, bool, bool) = &DoomGameLua::advanceAction;

double (DoomGameLua::*makeAction1)(object const&) = &DoomGameLua::makeAction;
double (DoomGameLua::*makeAction2)(object const&, unsigned int) = &DoomGameLua::makeAction;

#define ENUM_VAL_2_LUA(v) value( #v , v )
/* value("VALUE", VALUE) */


void sleepLua(unsigned int time){
    std::this_thread::sleep_for(std::chrono::milliseconds(time));
}

extern "C" int luaopen_vizdoom(lua_State *luaState){

    open(luaState);

    module(luaState, "vizdoom")[
    //module(luaState)[
        def("sleep", sleepLua),

        class_<int>("Mode")
            .enum_("Mode")[
                ENUM_VAL_2_LUA(PLAYER),
                ENUM_VAL_2_LUA(SPECTATOR),
                ENUM_VAL_2_LUA(ASYNC_PLAYER),
                ENUM_VAL_2_LUA(ASYNC_SPECTATOR)
            ],

        class_<int>("ScreenFormat")
            .enum_("ScreenFormat")[
                ENUM_VAL_2_LUA(CRCGCB),
                ENUM_VAL_2_LUA(RGB24),
                ENUM_VAL_2_LUA(RGBA32),
                ENUM_VAL_2_LUA(ARGB32),
                ENUM_VAL_2_LUA(CBCGCR),
                ENUM_VAL_2_LUA(BGR24),
                ENUM_VAL_2_LUA(BGRA32),
                ENUM_VAL_2_LUA(ABGR32),
                ENUM_VAL_2_LUA(GRAY8),
                ENUM_VAL_2_LUA(DOOM_256_COLORS8)
            ],

        class_<int>("ScreenResolution")
            .enum_("ScreenResolution")[
                ENUM_VAL_2_LUA(RES_160X120),

                ENUM_VAL_2_LUA(RES_200X125),
                ENUM_VAL_2_LUA(RES_200X150),

                ENUM_VAL_2_LUA(RES_256X144),
                ENUM_VAL_2_LUA(RES_256X160),
                ENUM_VAL_2_LUA(RES_256X192),

                ENUM_VAL_2_LUA(RES_320X180),
                ENUM_VAL_2_LUA(RES_320X200),
                ENUM_VAL_2_LUA(RES_320X240),
                ENUM_VAL_2_LUA(RES_320X256),

                ENUM_VAL_2_LUA(RES_400X225),
                ENUM_VAL_2_LUA(RES_400X250),
                ENUM_VAL_2_LUA(RES_400X300),

                ENUM_VAL_2_LUA(RES_512X288),
                ENUM_VAL_2_LUA(RES_512X320),
                ENUM_VAL_2_LUA(RES_512X384),

                ENUM_VAL_2_LUA(RES_640X360),
                ENUM_VAL_2_LUA(RES_640X400),
                ENUM_VAL_2_LUA(RES_640X480),

                ENUM_VAL_2_LUA(RES_800X450),
                ENUM_VAL_2_LUA(RES_800X500),
                ENUM_VAL_2_LUA(RES_800X600),

                ENUM_VAL_2_LUA(RES_1024X576),
                ENUM_VAL_2_LUA(RES_1024X640),
                ENUM_VAL_2_LUA(RES_1024X768),

                ENUM_VAL_2_LUA(RES_1280X720),
                ENUM_VAL_2_LUA(RES_1280X800),
                ENUM_VAL_2_LUA(RES_1280X960),
                ENUM_VAL_2_LUA(RES_1280X1024),

                ENUM_VAL_2_LUA(RES_1400X787),
                ENUM_VAL_2_LUA(RES_1400X875),
                ENUM_VAL_2_LUA(RES_1400X1050),

                ENUM_VAL_2_LUA(RES_1600X900),
                ENUM_VAL_2_LUA(RES_1600X1000),
                ENUM_VAL_2_LUA(RES_1600X1200),

                ENUM_VAL_2_LUA(RES_1920X1080)
            ],

        class_<int>("Button")
            .enum_("Button")[
                ENUM_VAL_2_LUA(ATTACK),
                ENUM_VAL_2_LUA(USE),
                ENUM_VAL_2_LUA(JUMP),
                ENUM_VAL_2_LUA(CROUCH),
                ENUM_VAL_2_LUA(TURN180),
                ENUM_VAL_2_LUA(ALTATTACK),
                ENUM_VAL_2_LUA(RELOAD),
                ENUM_VAL_2_LUA(ZOOM),
                ENUM_VAL_2_LUA(SPEED),
                ENUM_VAL_2_LUA(STRAFE),
                ENUM_VAL_2_LUA(MOVE_RIGHT),
                ENUM_VAL_2_LUA(MOVE_LEFT),
                ENUM_VAL_2_LUA(MOVE_BACKWARD),
                ENUM_VAL_2_LUA(MOVE_FORWARD),
                ENUM_VAL_2_LUA(TURN_RIGHT),
                ENUM_VAL_2_LUA(TURN_LEFT),
                ENUM_VAL_2_LUA(LOOK_UP),
                ENUM_VAL_2_LUA(LOOK_DOWN),
                ENUM_VAL_2_LUA(LAND),
                ENUM_VAL_2_LUA(SELECT_WEAPON1),
                ENUM_VAL_2_LUA(SELECT_WEAPON2),
                ENUM_VAL_2_LUA(SELECT_WEAPON3),
                ENUM_VAL_2_LUA(SELECT_WEAPON4),
                ENUM_VAL_2_LUA(SELECT_WEAPON5),
                ENUM_VAL_2_LUA(SELECT_WEAPON6),
                ENUM_VAL_2_LUA(SELECT_WEAPON7),
                ENUM_VAL_2_LUA(SELECT_WEAPON8),
                ENUM_VAL_2_LUA(SELECT_WEAPON9),
                ENUM_VAL_2_LUA(SELECT_WEAPON0),
                ENUM_VAL_2_LUA(SELECT_NEXT_WEAPON),
                ENUM_VAL_2_LUA(SELECT_PREV_WEAPON),
                ENUM_VAL_2_LUA(DROP_SELECTED_WEAPON),
                ENUM_VAL_2_LUA(ACTIVATE_SELECTED_ITEM),
                ENUM_VAL_2_LUA(SELECT_NEXT_ITEM),
                ENUM_VAL_2_LUA(SELECT_PREV_ITEM),
                ENUM_VAL_2_LUA(DROP_SELECTED_ITEM),
                ENUM_VAL_2_LUA(LOOK_UP_DOWN_DELTA),
                ENUM_VAL_2_LUA(TURN_LEFT_RIGHT_DELTA),
                ENUM_VAL_2_LUA(MOVE_FORWARD_BACKWARD_DELTA),
                ENUM_VAL_2_LUA(MOVE_LEFT_RIGHT_DELTA),
                ENUM_VAL_2_LUA(MOVE_UP_DOWN_DELTA)
            ],

        class_<int>("GameVariable")
            .enum_("GameVariable")[
                ENUM_VAL_2_LUA(KILLCOUNT),
                ENUM_VAL_2_LUA(ITEMCOUNT),
                ENUM_VAL_2_LUA(SECRETCOUNT),
                ENUM_VAL_2_LUA(FRAGCOUNT),
                ENUM_VAL_2_LUA(DEATHCOUNT),
                ENUM_VAL_2_LUA(HEALTH),
                ENUM_VAL_2_LUA(ARMOR),
                ENUM_VAL_2_LUA(DEAD),
                ENUM_VAL_2_LUA(ON_GROUND),
                ENUM_VAL_2_LUA(ATTACK_READY),
                ENUM_VAL_2_LUA(ALTATTACK_READY),
                ENUM_VAL_2_LUA(SELECTED_WEAPON),
                ENUM_VAL_2_LUA(SELECTED_WEAPON_AMMO),
                ENUM_VAL_2_LUA(AMMO1),
                ENUM_VAL_2_LUA(AMMO2),
                ENUM_VAL_2_LUA(AMMO3),
                ENUM_VAL_2_LUA(AMMO4),
                ENUM_VAL_2_LUA(AMMO5),
                ENUM_VAL_2_LUA(AMMO6),
                ENUM_VAL_2_LUA(AMMO7),
                ENUM_VAL_2_LUA(AMMO8),
                ENUM_VAL_2_LUA(AMMO9),
                ENUM_VAL_2_LUA(AMMO0),
                ENUM_VAL_2_LUA(WEAPON1),
                ENUM_VAL_2_LUA(WEAPON2),
                ENUM_VAL_2_LUA(WEAPON3),
                ENUM_VAL_2_LUA(WEAPON4),
                ENUM_VAL_2_LUA(WEAPON5),
                ENUM_VAL_2_LUA(WEAPON6),
                ENUM_VAL_2_LUA(WEAPON7),
                ENUM_VAL_2_LUA(WEAPON8),
                ENUM_VAL_2_LUA(WEAPON9),
                ENUM_VAL_2_LUA(WEAPON0),
                ENUM_VAL_2_LUA(USER1),
                ENUM_VAL_2_LUA(USER2),
                ENUM_VAL_2_LUA(USER3),
                ENUM_VAL_2_LUA(USER4),
                ENUM_VAL_2_LUA(USER5),
                ENUM_VAL_2_LUA(USER6),
                ENUM_VAL_2_LUA(USER7),
                ENUM_VAL_2_LUA(USER8),
                ENUM_VAL_2_LUA(USER9),
                ENUM_VAL_2_LUA(USER10),
                ENUM_VAL_2_LUA(USER11),
                ENUM_VAL_2_LUA(USER12),
                ENUM_VAL_2_LUA(USER13),
                ENUM_VAL_2_LUA(USER14),
                ENUM_VAL_2_LUA(USER15),
                ENUM_VAL_2_LUA(USER16),
                ENUM_VAL_2_LUA(USER17),
                ENUM_VAL_2_LUA(USER18),
                ENUM_VAL_2_LUA(USER19),
                ENUM_VAL_2_LUA(USER20),
                ENUM_VAL_2_LUA(USER21),
                ENUM_VAL_2_LUA(USER22),
                ENUM_VAL_2_LUA(USER23),
                ENUM_VAL_2_LUA(USER24),
                ENUM_VAL_2_LUA(USER25),
                ENUM_VAL_2_LUA(USER26),
                ENUM_VAL_2_LUA(USER27),
                ENUM_VAL_2_LUA(USER28),
                ENUM_VAL_2_LUA(USER29),
                ENUM_VAL_2_LUA(USER30),
                ENUM_VAL_2_LUA(PLAYER_NUMBER),
                ENUM_VAL_2_LUA(PLAYER_COUNT),
                ENUM_VAL_2_LUA(PLAYER1_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER2_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER3_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER4_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER5_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER6_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER7_FRAGCOUNT),
                ENUM_VAL_2_LUA(PLAYER8_FRAGCOUNT)
            ],

        class_<DoomGameLua>("DoomGame")
            .def(constructor<>())
            .def("init", &DoomGameLua::init)
            .def("load_config", &DoomGameLua::loadConfig)
            .def("close", &DoomGameLua::close)
            .def("new_episode", newEpisode1)
            .def("new_episode", newEpisode2)
            .def("replay_episode", &DoomGameLua::replayEpisode)
            .def("is_episode_finished", &DoomGameLua::isEpisodeFinished)
            .def("is_new_episode", &DoomGameLua::isNewEpisode)
            .def("is_player_dead", &DoomGameLua::isPlayerDead)
            .def("respawn_player", &DoomGameLua::respawnPlayer)
            .def("set_action", &DoomGameLua::setAction)
            .def("make_action", makeAction1)
            .def("make_action", makeAction2)
            .def("advance_action", advanceAction1)
            .def("advance_action", advanceAction2)
            .def("advance_action", advanceAction3)

            .def("get_state", &DoomGameLua::getState)

            .def("get_game_variable", &DoomGameLua::getGameVariable)

            .def("get_living_reward", &DoomGameLua::getLivingReward)
            .def("set_living_reward", &DoomGameLua::setLivingReward)

            .def("get_death_penalty", &DoomGameLua::getDeathPenalty)
            .def("set_death_penalty", &DoomGameLua::setDeathPenalty)

            .def("get_last_reward", &DoomGameLua::getLastReward)
            .def("get_total_reward", &DoomGameLua::getTotalReward)

            .def("get_last_action", &DoomGameLua::getLastAction)

            .def("add_available_game_variable", &DoomGameLua::addAvailableGameVariable)
            .def("clear_available_game_variables", &DoomGameLua::clearAvailableGameVariables)
            .def("get_available_game_variables_size", &DoomGameLua::getAvailableGameVariablesSize)

            .def("add_available_button", addAvailableButton1)
            .def("add_available_button", addAvailableButton2)

            .def("clear_available_buttons", &DoomGameLua::clearAvailableButtons)
            .def("get_available_buttons_size", &DoomGameLua::getAvailableButtonsSize)
            .def("set_button_max_value", &DoomGameLua::setButtonMaxValue)
            .def("get_button_max_value", &DoomGameLua::getButtonMaxValue)

            .def("add_game_args", &DoomGameLua::addGameArgs)
            .def("clear_game_args", &DoomGameLua::clearGameArgs)

            .def("send_game_command", &DoomGameLua::sendGameCommand)

            .def("get_mode", &DoomGameLua::getMode)
            .def("set_mode", &DoomGameLua::setMode)

            .def("get_ticrate", &DoomGameLua::getTicrate)
            .def("set_ticrate", &DoomGameLua::setTicrate)

            .def("set_vizdoom_path", &DoomGameLua::setViZDoomPath)
            .def("set_doom_game_path", &DoomGameLua::setDoomGamePath)
            .def("set_doom_scenario_path", &DoomGameLua::setDoomScenarioPath)
            .def("set_doom_map", &DoomGameLua::setDoomMap)
            .def("set_doom_skill", &DoomGameLua::setDoomSkill)
            .def("set_doom_config_path", &DoomGameLua::setDoomConfigPath)

            .def("get_seed", &DoomGameLua::getSeed)
            .def("set_seed", &DoomGameLua::setSeed)

            .def("get_episode_start_time", &DoomGameLua::getEpisodeStartTime)
            .def("set_episode_start_time", &DoomGameLua::setEpisodeStartTime)
            .def("get_episode_timeout", &DoomGameLua::getEpisodeTimeout)
            .def("set_episode_timeout", &DoomGameLua::setEpisodeTimeout)
            .def("get_episode_time", &DoomGameLua::getEpisodeTime)

            .def("set_console_enabled",&DoomGameLua::setConsoleEnabled)
            .def("set_sound_enabled",&DoomGameLua::setSoundEnabled)

            .def("set_screen_resolution", &DoomGameLua::setScreenResolution)
            .def("set_screen_format", &DoomGameLua::setScreenFormat)

            .def("set_depth_buffer_enabled", &DoomGameLua::setDepthBufferEnabled)
            .def("set_labels_buffer_enabled", &DoomGameLua::setLabelsBufferEnabled)
            .def("set_map_buffer_enabled", &DoomGameLua::setMapBufferEnabled)

            .def("set_render_hud", &DoomGameLua::setRenderHud)
            .def("set_render_weapon", &DoomGameLua::setRenderWeapon)
            .def("set_render_crosshair", &DoomGameLua::setRenderCrosshair)
            .def("set_render_decals", &DoomGameLua::setRenderDecals)
            .def("set_render_particles", &DoomGameLua::setRenderParticles)
            .def("get_screen_width", &DoomGameLua::getScreenWidth)
            .def("get_screen_height", &DoomGameLua::getScreenHeight)
            .def("get_screen_channels", &DoomGameLua::getScreenChannels)
            .def("get_screen_size", &DoomGameLua::getScreenSize)
            .def("get_screen_pitch", &DoomGameLua::getScreenPitch)
            .def("get_screen_format", &DoomGameLua::getScreenFormat)
            .def("set_window_visible", &DoomGameLua::setWindowVisible),

        class_<GameStateLua>("GameState")
            .def_readonly("number", &GameStateLua::number)
            .def_readonly("game_variables", &GameStateLua::gameVariables)
            .def_readonly("screen_buffer", &GameStateLua::screenBuffer)
            .def_readonly("depth_buffer", &GameStateLua::depthBuffer)
            .def_readonly("map_buffer", &GameStateLua::mapBuffer)
            .def_readonly("labels_buffer", &GameStateLua::labelsBuffer),

        class_<LabelLua>("Label")
            .def_readonly("object_id", &LabelLua::objectId)
            .def_readonly("object_name", &LabelLua::objectName)
            .def_readonly("value", &LabelLua::value)
    ];

    return 1;
}