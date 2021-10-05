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

#include "ViZDoom.h"
#include "ViZDoomGamePython.h"
#include "ViZDoomVersion.h"

#include <pybind11/pybind11.h>
#include <vector>

using namespace vizdoom;

/* C++ code to expose DoomGame library via Python */


/* Overloaded functions */
/*--------------------------------------------------------------------------------------------------------------------*/

double (*doomFixedToDouble_int)(int) = &doomFixedToDouble;
double (*doomFixedToDouble_double)(double) = &doomFixedToDouble;


/* Module definition */
/*--------------------------------------------------------------------------------------------------------------------*/

PYBIND11_MODULE(vizdoom, vz){

    using namespace pybind11;
    vz.attr("__version__") = pyb::str(VIZDOOM_LIB_VERSION_STR);

    Py_Initialize();
    PyEval_InitThreads();

    /* Exceptions */
    /*----------------------------------------------------------------------------------------------------------------*/

    #define EXCEPTION_TO_PYT(n) register_exception< n >(vz, #n);
    /* register_exception< ExceptionName >(vz, "ExceptionName"); */

    EXCEPTION_TO_PYT(FileDoesNotExistException)
    EXCEPTION_TO_PYT(MessageQueueException)
    EXCEPTION_TO_PYT(SharedMemoryException)
    EXCEPTION_TO_PYT(SignalException)
    EXCEPTION_TO_PYT(ViZDoomIsNotRunningException)
    EXCEPTION_TO_PYT(ViZDoomErrorException)
    EXCEPTION_TO_PYT(ViZDoomUnexpectedExitException)


    /* Helpers */
    /*----------------------------------------------------------------------------------------------------------------*/

    #define CONST_2_PYT(c) vz.attr( #c ) = c
    /* vz.attr("CONST") = CONST */

    #define ENUM_VAL_2_PYT(v) .value( #v , v )
    /* .value("VALUE", VALUE) */

    #define ENUM_CLASS_VAL_2_PYT(c, v) .value( #v , c::v )
    /* .value("VALUE", class::VALUE) */

    #define FUNC_2_PYT(f) vz.def( #f , f )
    /* vz.def("function", function) */

    #define CLASS_FUNC_2_PYT(c, f) .def( #f , &c::f )
    /* .def("function", &class::function) */


    /* Consts */
    /*----------------------------------------------------------------------------------------------------------------*/

    CONST_2_PYT(SLOT_COUNT);
    CONST_2_PYT(MAX_PLAYERS);
    CONST_2_PYT(MAX_PLAYER_NAME_LENGTH);
    CONST_2_PYT(USER_VARIABLE_COUNT);
    CONST_2_PYT(DEFAULT_TICRATE);

    CONST_2_PYT(BINARY_BUTTON_COUNT);
    CONST_2_PYT(DELTA_BUTTON_COUNT);
    CONST_2_PYT(BUTTON_COUNT);


    /* Enums */
    /*----------------------------------------------------------------------------------------------------------------*/

    enum_<Mode>(vz, "Mode")
        ENUM_VAL_2_PYT(PLAYER)
        ENUM_VAL_2_PYT(SPECTATOR)
        ENUM_VAL_2_PYT(ASYNC_PLAYER)
        ENUM_VAL_2_PYT(ASYNC_SPECTATOR);

    enum_<ScreenFormat>(vz, "ScreenFormat")
        ENUM_VAL_2_PYT(CRCGCB)
        ENUM_VAL_2_PYT(RGB24)
        ENUM_VAL_2_PYT(RGBA32)
        ENUM_VAL_2_PYT(ARGB32)
        ENUM_VAL_2_PYT(CBCGCR)
        ENUM_VAL_2_PYT(BGR24)
        ENUM_VAL_2_PYT(BGRA32)
        ENUM_VAL_2_PYT(ABGR32)
        ENUM_VAL_2_PYT(GRAY8)
        ENUM_VAL_2_PYT(DOOM_256_COLORS8);

    enum_<ScreenResolution>(vz, "ScreenResolution")
        ENUM_VAL_2_PYT(RES_160X120)

        ENUM_VAL_2_PYT(RES_200X125)
        ENUM_VAL_2_PYT(RES_200X150)

        ENUM_VAL_2_PYT(RES_256X144)
        ENUM_VAL_2_PYT(RES_256X160)
        ENUM_VAL_2_PYT(RES_256X192)

        ENUM_VAL_2_PYT(RES_320X180)
        ENUM_VAL_2_PYT(RES_320X200)
        ENUM_VAL_2_PYT(RES_320X240)
        ENUM_VAL_2_PYT(RES_320X256)

        ENUM_VAL_2_PYT(RES_400X225)
        ENUM_VAL_2_PYT(RES_400X250)
        ENUM_VAL_2_PYT(RES_400X300)

        ENUM_VAL_2_PYT(RES_512X288)
        ENUM_VAL_2_PYT(RES_512X320)
        ENUM_VAL_2_PYT(RES_512X384)

        ENUM_VAL_2_PYT(RES_640X360)
        ENUM_VAL_2_PYT(RES_640X400)
        ENUM_VAL_2_PYT(RES_640X480)

        ENUM_VAL_2_PYT(RES_800X450)
        ENUM_VAL_2_PYT(RES_800X500)
        ENUM_VAL_2_PYT(RES_800X600)

        ENUM_VAL_2_PYT(RES_1024X576)
        ENUM_VAL_2_PYT(RES_1024X640)
        ENUM_VAL_2_PYT(RES_1024X768)

        ENUM_VAL_2_PYT(RES_1280X720)
        ENUM_VAL_2_PYT(RES_1280X800)
        ENUM_VAL_2_PYT(RES_1280X960)
        ENUM_VAL_2_PYT(RES_1280X1024)

        ENUM_VAL_2_PYT(RES_1400X787)
        ENUM_VAL_2_PYT(RES_1400X875)
        ENUM_VAL_2_PYT(RES_1400X1050)

        ENUM_VAL_2_PYT(RES_1600X900)
        ENUM_VAL_2_PYT(RES_1600X1000)
        ENUM_VAL_2_PYT(RES_1600X1200)

        ENUM_VAL_2_PYT(RES_1920X1080)
        .export_values();

    enum_<AutomapMode>(vz, "AutomapMode")
        ENUM_VAL_2_PYT(NORMAL)
        ENUM_VAL_2_PYT(WHOLE)
        ENUM_VAL_2_PYT(OBJECTS)
        ENUM_VAL_2_PYT(OBJECTS_WITH_SIZE)
        .export_values();

    enum_<Button>(vz, "Button")
        ENUM_VAL_2_PYT(ATTACK)
        ENUM_VAL_2_PYT(USE)
        ENUM_VAL_2_PYT(JUMP)
        ENUM_VAL_2_PYT(CROUCH)
        ENUM_VAL_2_PYT(TURN180)
        ENUM_VAL_2_PYT(ALTATTACK)
        ENUM_VAL_2_PYT(RELOAD)
        ENUM_VAL_2_PYT(ZOOM)
        ENUM_VAL_2_PYT(SPEED)
        ENUM_VAL_2_PYT(STRAFE)
        ENUM_VAL_2_PYT(MOVE_RIGHT)
        ENUM_VAL_2_PYT(MOVE_LEFT)
        ENUM_VAL_2_PYT(MOVE_BACKWARD)
        ENUM_VAL_2_PYT(MOVE_FORWARD)
        ENUM_VAL_2_PYT(TURN_RIGHT)
        ENUM_VAL_2_PYT(TURN_LEFT)
        ENUM_VAL_2_PYT(LOOK_UP)
        ENUM_VAL_2_PYT(LOOK_DOWN)
        ENUM_VAL_2_PYT(MOVE_UP)
        ENUM_VAL_2_PYT(MOVE_DOWN)
        ENUM_VAL_2_PYT(LAND)
        ENUM_VAL_2_PYT(SELECT_WEAPON1)
        ENUM_VAL_2_PYT(SELECT_WEAPON2)
        ENUM_VAL_2_PYT(SELECT_WEAPON3)
        ENUM_VAL_2_PYT(SELECT_WEAPON4)
        ENUM_VAL_2_PYT(SELECT_WEAPON5)
        ENUM_VAL_2_PYT(SELECT_WEAPON6)
        ENUM_VAL_2_PYT(SELECT_WEAPON7)
        ENUM_VAL_2_PYT(SELECT_WEAPON8)
        ENUM_VAL_2_PYT(SELECT_WEAPON9)
        ENUM_VAL_2_PYT(SELECT_WEAPON0)
        ENUM_VAL_2_PYT(SELECT_NEXT_WEAPON)
        ENUM_VAL_2_PYT(SELECT_PREV_WEAPON)
        ENUM_VAL_2_PYT(DROP_SELECTED_WEAPON)
        ENUM_VAL_2_PYT(ACTIVATE_SELECTED_ITEM)
        ENUM_VAL_2_PYT(SELECT_NEXT_ITEM)
        ENUM_VAL_2_PYT(SELECT_PREV_ITEM)
        ENUM_VAL_2_PYT(DROP_SELECTED_ITEM)
        ENUM_VAL_2_PYT(LOOK_UP_DOWN_DELTA)
        ENUM_VAL_2_PYT(TURN_LEFT_RIGHT_DELTA)
        ENUM_VAL_2_PYT(MOVE_FORWARD_BACKWARD_DELTA)
        ENUM_VAL_2_PYT(MOVE_LEFT_RIGHT_DELTA)
        ENUM_VAL_2_PYT(MOVE_UP_DOWN_DELTA)
        .export_values();

    enum_<GameVariable>(vz, "GameVariable")
        ENUM_VAL_2_PYT(KILLCOUNT)
        ENUM_VAL_2_PYT(ITEMCOUNT)
        ENUM_VAL_2_PYT(SECRETCOUNT)
        ENUM_VAL_2_PYT(FRAGCOUNT)
        ENUM_VAL_2_PYT(DEATHCOUNT)
        ENUM_VAL_2_PYT(HITCOUNT)
        ENUM_VAL_2_PYT(HITS_TAKEN)
        ENUM_VAL_2_PYT(DAMAGECOUNT)
        ENUM_VAL_2_PYT(DAMAGE_TAKEN)
        ENUM_VAL_2_PYT(HEALTH)
        ENUM_VAL_2_PYT(ARMOR)
        ENUM_VAL_2_PYT(DEAD)
        ENUM_VAL_2_PYT(ON_GROUND)
        ENUM_VAL_2_PYT(ATTACK_READY)
        ENUM_VAL_2_PYT(ALTATTACK_READY)
        ENUM_VAL_2_PYT(SELECTED_WEAPON)
        ENUM_VAL_2_PYT(SELECTED_WEAPON_AMMO)
        ENUM_VAL_2_PYT(AMMO1)
        ENUM_VAL_2_PYT(AMMO2)
        ENUM_VAL_2_PYT(AMMO3)
        ENUM_VAL_2_PYT(AMMO4)
        ENUM_VAL_2_PYT(AMMO5)
        ENUM_VAL_2_PYT(AMMO6)
        ENUM_VAL_2_PYT(AMMO7)
        ENUM_VAL_2_PYT(AMMO8)
        ENUM_VAL_2_PYT(AMMO9)
        ENUM_VAL_2_PYT(AMMO0)
        ENUM_VAL_2_PYT(WEAPON1)
        ENUM_VAL_2_PYT(WEAPON2)
        ENUM_VAL_2_PYT(WEAPON3)
        ENUM_VAL_2_PYT(WEAPON4)
        ENUM_VAL_2_PYT(WEAPON5)
        ENUM_VAL_2_PYT(WEAPON6)
        ENUM_VAL_2_PYT(WEAPON7)
        ENUM_VAL_2_PYT(WEAPON8)
        ENUM_VAL_2_PYT(WEAPON9)
        ENUM_VAL_2_PYT(WEAPON0)
        ENUM_VAL_2_PYT(POSITION_X)
        ENUM_VAL_2_PYT(POSITION_Y)
        ENUM_VAL_2_PYT(POSITION_Z)
        ENUM_VAL_2_PYT(ANGLE)
        ENUM_VAL_2_PYT(PITCH)
        ENUM_VAL_2_PYT(ROLL)
        ENUM_VAL_2_PYT(VIEW_HEIGHT)
        ENUM_VAL_2_PYT(VELOCITY_X)
        ENUM_VAL_2_PYT(VELOCITY_Y)
        ENUM_VAL_2_PYT(VELOCITY_Z)
        ENUM_VAL_2_PYT(CAMERA_POSITION_X)
        ENUM_VAL_2_PYT(CAMERA_POSITION_Y)
        ENUM_VAL_2_PYT(CAMERA_POSITION_Z)
        ENUM_VAL_2_PYT(CAMERA_ANGLE)
        ENUM_VAL_2_PYT(CAMERA_PITCH)
        ENUM_VAL_2_PYT(CAMERA_ROLL)
        ENUM_VAL_2_PYT(CAMERA_FOV)
        ENUM_VAL_2_PYT(USER1)
        ENUM_VAL_2_PYT(USER2)
        ENUM_VAL_2_PYT(USER3)
        ENUM_VAL_2_PYT(USER4)
        ENUM_VAL_2_PYT(USER5)
        ENUM_VAL_2_PYT(USER6)
        ENUM_VAL_2_PYT(USER7)
        ENUM_VAL_2_PYT(USER8)
        ENUM_VAL_2_PYT(USER9)
        ENUM_VAL_2_PYT(USER10)
        ENUM_VAL_2_PYT(USER11)
        ENUM_VAL_2_PYT(USER12)
        ENUM_VAL_2_PYT(USER13)
        ENUM_VAL_2_PYT(USER14)
        ENUM_VAL_2_PYT(USER15)
        ENUM_VAL_2_PYT(USER16)
        ENUM_VAL_2_PYT(USER17)
        ENUM_VAL_2_PYT(USER18)
        ENUM_VAL_2_PYT(USER19)
        ENUM_VAL_2_PYT(USER20)
        ENUM_VAL_2_PYT(USER21)
        ENUM_VAL_2_PYT(USER22)
        ENUM_VAL_2_PYT(USER23)
        ENUM_VAL_2_PYT(USER24)
        ENUM_VAL_2_PYT(USER25)
        ENUM_VAL_2_PYT(USER26)
        ENUM_VAL_2_PYT(USER27)
        ENUM_VAL_2_PYT(USER28)
        ENUM_VAL_2_PYT(USER29)
        ENUM_VAL_2_PYT(USER30)
        ENUM_VAL_2_PYT(USER31)
        ENUM_VAL_2_PYT(USER32)
        ENUM_VAL_2_PYT(USER33)
        ENUM_VAL_2_PYT(USER34)
        ENUM_VAL_2_PYT(USER35)
        ENUM_VAL_2_PYT(USER36)
        ENUM_VAL_2_PYT(USER37)
        ENUM_VAL_2_PYT(USER38)
        ENUM_VAL_2_PYT(USER39)
        ENUM_VAL_2_PYT(USER40)
        ENUM_VAL_2_PYT(USER41)
        ENUM_VAL_2_PYT(USER42)
        ENUM_VAL_2_PYT(USER43)
        ENUM_VAL_2_PYT(USER44)
        ENUM_VAL_2_PYT(USER45)
        ENUM_VAL_2_PYT(USER46)
        ENUM_VAL_2_PYT(USER47)
        ENUM_VAL_2_PYT(USER48)
        ENUM_VAL_2_PYT(USER49)
        ENUM_VAL_2_PYT(USER50)
        ENUM_VAL_2_PYT(USER51)
        ENUM_VAL_2_PYT(USER52)
        ENUM_VAL_2_PYT(USER53)
        ENUM_VAL_2_PYT(USER54)
        ENUM_VAL_2_PYT(USER55)
        ENUM_VAL_2_PYT(USER56)
        ENUM_VAL_2_PYT(USER57)
        ENUM_VAL_2_PYT(USER58)
        ENUM_VAL_2_PYT(USER59)
        ENUM_VAL_2_PYT(USER60)
        ENUM_VAL_2_PYT(PLAYER_NUMBER)
        ENUM_VAL_2_PYT(PLAYER_COUNT)
        ENUM_VAL_2_PYT(PLAYER1_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER2_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER3_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER4_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER5_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER6_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER7_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER8_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER9_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER10_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER11_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER12_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER13_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER14_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER15_FRAGCOUNT)
        ENUM_VAL_2_PYT(PLAYER16_FRAGCOUNT)
        .export_values();

    enum_<SamplingRate>(vz, "SamplingRate")
        ENUM_VAL_2_PYT(SR_11025)
        ENUM_VAL_2_PYT(SR_22050)
        ENUM_VAL_2_PYT(SR_44100)
        .export_values();

    /* Structs */
    /*----------------------------------------------------------------------------------------------------------------*/

    #define LABEL_CLASS Label
    class_<LABEL_CLASS>(vz, "Label")
        .def_readonly("object_id", &LABEL_CLASS::objectId)
        .def_readonly("object_name", &LABEL_CLASS::objectName)
        .def_readonly("value", &LABEL_CLASS::value)
        .def_readonly("x", &LABEL_CLASS::x)
        .def_readonly("y", &LABEL_CLASS::y)
        .def_readonly("width", &LABEL_CLASS::width)
        .def_readonly("height", &LABEL_CLASS::height)
        .def_readonly("object_position_x", &LABEL_CLASS::objectPositionX)
        .def_readonly("object_position_y", &LABEL_CLASS::objectPositionY)
        .def_readonly("object_position_z", &LABEL_CLASS::objectPositionZ)
        .def_readonly("object_angle", &LABEL_CLASS::objectAngle)
        .def_readonly("object_pitch", &LABEL_CLASS::objectPitch)
        .def_readonly("object_roll", &LABEL_CLASS::objectRoll)
        .def_readonly("object_velocity_x", &LABEL_CLASS::objectVelocityX)
        .def_readonly("object_velocity_y", &LABEL_CLASS::objectVelocityY)
        .def_readonly("object_velocity_z", &LABEL_CLASS::objectVelocityZ);

    #define OBJECT_CLASS Object
    class_<OBJECT_CLASS>(vz, "Object")
        .def_readonly("id", &OBJECT_CLASS::id)
        .def_readonly("name", &OBJECT_CLASS::name)
        .def_readonly("position_x", &OBJECT_CLASS::positionX)
        .def_readonly("position_y", &OBJECT_CLASS::positionY)
        .def_readonly("position_z", &OBJECT_CLASS::positionZ)
        .def_readonly("angle", &OBJECT_CLASS::angle)
        .def_readonly("pitch", &OBJECT_CLASS::pitch)
        .def_readonly("roll", &OBJECT_CLASS::roll)
        .def_readonly("velocity_x", &OBJECT_CLASS::velocityX)
        .def_readonly("velocity_y", &OBJECT_CLASS::velocityY)
        .def_readonly("velocity_z", &OBJECT_CLASS::velocityZ);

    #define LINE_CLASS Line
    class_<LINE_CLASS>(vz, "Line")
        .def_readonly("x1", &LINE_CLASS::x1)
        .def_readonly("y1", &LINE_CLASS::y1)
        .def_readonly("x2", &LINE_CLASS::x2)
        .def_readonly("y2", &LINE_CLASS::y2)
        .def_readonly("is_blocking", &LINE_CLASS::isBlocking);

    #define SECTOR_CLASS SectorPython
    class_<SECTOR_CLASS>(vz, "Sector")
        .def_readonly("floor_height", &SECTOR_CLASS::floorHeight)
        .def_readonly("ceiling_height", &SECTOR_CLASS::ceilingHeight)
        .def_readonly("lines", &SECTOR_CLASS::lines);

    class_<GameStatePython>(vz, "GameState")
        .def_readonly("number", &GameStatePython::number)
        .def_readonly("tic", &GameStatePython::tic)
        .def_readonly("game_variables", &GameStatePython::gameVariables)

        .def_readonly("screen_buffer", &GameStatePython::screenBuffer)
        .def_readonly("audio_buffer", &GameStatePython::audioBuffer)
        .def_readonly("depth_buffer", &GameStatePython::depthBuffer)
        .def_readonly("labels_buffer", &GameStatePython::labelsBuffer)
        .def_readonly("automap_buffer", &GameStatePython::automapBuffer)

        .def_readonly("labels", &GameStatePython::labels)
        .def_readonly("objects", &GameStatePython::objects)
        .def_readonly("sectors", &GameStatePython::sectors);

    class_<ServerStatePython>(vz, "ServerState")
        .def_readonly("tic", &ServerStatePython::tic)
        .def_readonly("player_count", &ServerStatePython::playerCount)
        .def_readonly("players_in_game", &ServerStatePython::playersInGame)
        .def_readonly("players_names", &ServerStatePython::playersNames)
        .def_readonly("players_frags", &ServerStatePython::playersFrags)
        .def_readonly("players_afk", &ServerStatePython::playersAfk)
        .def_readonly("players_last_action_tic", &ServerStatePython::playersLastActionTic)
        .def_readonly("players_last_kill_tic", &ServerStatePython::playersLastKillTic);


    /* DoomGame */
    /*----------------------------------------------------------------------------------------------------------------*/

    class_<DoomGamePython>(vz, "DoomGame")
        .def(init<>())
        .def("init", &DoomGamePython::init)
        .def("load_config", &DoomGamePython::loadConfig)
        .def("close", &DoomGamePython::close)
        .def("is_running", &DoomGamePython::isRunning)
        .def("is_multiplayer_game", &DoomGamePython::isMultiplayerGame)
        .def("is_recording_episode", &DoomGamePython::isRecordingEpisode)
        .def("is_replaying_episode", &DoomGamePython::isReplayingEpisode)
        .def("new_episode", &DoomGamePython::newEpisode_)
        .def("new_episode", &DoomGamePython::newEpisode_str)
        .def("replay_episode", &DoomGamePython::replayEpisode_str)
        .def("replay_episode", &DoomGamePython::replayEpisode_str_int)
        .def("is_episode_finished", &DoomGamePython::isEpisodeFinished)
        .def("is_new_episode", &DoomGamePython::isNewEpisode)
        .def("is_player_dead", &DoomGamePython::isPlayerDead)
        .def("respawn_player", &DoomGamePython::respawnPlayer)
        .def("set_action", &DoomGamePython::setAction)
        .def("make_action", &DoomGamePython::makeAction_list)
        .def("make_action", &DoomGamePython::makeAction_list_int)
        .def("advance_action", &DoomGamePython::advanceAction_)
        .def("advance_action", &DoomGamePython::advanceAction_int)
        .def("advance_action", &DoomGamePython::advanceAction_int_bool)
        .def("save", &DoomGamePython::save)
        .def("load", &DoomGamePython::load)

        .def("get_state", &DoomGamePython::getState, return_value_policy::take_ownership)
        .def("get_server_state", &DoomGamePython::getServerState, return_value_policy::take_ownership)

        .def("get_game_variable", &DoomGamePython::getGameVariable)
        .def("get_button", &DoomGamePython::getButton)

        .def("get_living_reward", &DoomGamePython::getLivingReward)
        .def("set_living_reward", &DoomGamePython::setLivingReward)

        .def("get_death_penalty", &DoomGamePython::getDeathPenalty)
        .def("set_death_penalty", &DoomGamePython::setDeathPenalty)

        .def("get_last_reward", &DoomGamePython::getLastReward)
        .def("get_total_reward", &DoomGamePython::getTotalReward)

        .def("get_last_action", &DoomGamePython::getLastAction)

        .def("get_available_game_variables", &DoomGamePython::getAvailableGameVariables)
        .def("set_available_game_variables", &DoomGamePython::setAvailableGameVariables)
        .def("add_available_game_variable", &DoomGamePython::addAvailableGameVariable)
        .def("clear_available_game_variables", &DoomGamePython::clearAvailableGameVariables)
        .def("get_available_game_variables_size", &DoomGamePython::getAvailableGameVariablesSize)

        .def("get_available_buttons", &DoomGamePython::getAvailableButtons)
        .def("set_available_buttons", &DoomGamePython::setAvailableButtons)
        .def("add_available_button", &DoomGamePython::addAvailableButton_btn)
        .def("add_available_button", &DoomGamePython::addAvailableButton_btn_int)
        .def("clear_available_buttons", &DoomGamePython::clearAvailableButtons)
        .def("get_available_buttons_size", &DoomGamePython::getAvailableButtonsSize)
        .def("set_button_max_value", &DoomGamePython::setButtonMaxValue)
        .def("get_button_max_value", &DoomGamePython::getButtonMaxValue)

        .def("add_game_args", &DoomGamePython::addGameArgs)
        .def("clear_game_args", &DoomGamePython::clearGameArgs)

        .def("send_game_command", &DoomGamePython::sendGameCommand)

        .def("get_mode", &DoomGamePython::getMode)
        .def("set_mode", &DoomGamePython::setMode)

        .def("get_ticrate", &DoomGamePython::getTicrate)
        .def("set_ticrate", &DoomGamePython::setTicrate)

        .def("set_vizdoom_path", &DoomGamePython::setViZDoomPath)
        .def("set_doom_game_path", &DoomGamePython::setDoomGamePath)
        .def("set_doom_scenario_path", &DoomGamePython::setDoomScenarioPath)
        .def("set_doom_map", &DoomGamePython::setDoomMap)
        .def("set_doom_skill", &DoomGamePython::setDoomSkill)
        .def("set_doom_config_path", &DoomGamePython::setDoomConfigPath)

        .def("get_seed", &DoomGamePython::getSeed)
        .def("set_seed", &DoomGamePython::setSeed)

        .def("get_episode_start_time", &DoomGamePython::getEpisodeStartTime)
        .def("set_episode_start_time", &DoomGamePython::setEpisodeStartTime)
        .def("get_episode_timeout", &DoomGamePython::getEpisodeTimeout)
        .def("set_episode_timeout", &DoomGamePython::setEpisodeTimeout)
        .def("get_episode_time", &DoomGamePython::getEpisodeTime)

        .def("set_console_enabled",&DoomGamePython::setConsoleEnabled)
        .def("set_sound_enabled",&DoomGamePython::setSoundEnabled)

        .def("is_audio_buffer_enabled",&DoomGamePython::isAudioBufferEnabled)
        .def("set_audio_buffer_enabled",&DoomGamePython::setAudioBufferEnabled)
        .def("get_audio_sampling_rate",&DoomGamePython::getAudioSamplingRate)
        .def("set_audio_sampling_rate",&DoomGamePython::setAudioSamplingRate)
        .def("get_audio_buffer_size",&DoomGamePython::getAudioBufferSize)
        .def("set_audio_buffer_size",&DoomGamePython::setAudioBufferSize)

        .def("set_screen_resolution", &DoomGamePython::setScreenResolution)
        .def("set_screen_format", &DoomGamePython::setScreenFormat)

        .def("is_depth_buffer_enabled", &DoomGamePython::isDepthBufferEnabled)
        .def("set_depth_buffer_enabled", &DoomGamePython::setDepthBufferEnabled)
        .def("is_labels_buffer_enabled", &DoomGamePython::isLabelsBufferEnabled)
        .def("set_labels_buffer_enabled", &DoomGamePython::setLabelsBufferEnabled)
        .def("is_automap_buffer_enabled", &DoomGamePython::isAutomapBufferEnabled)
        .def("set_automap_buffer_enabled", &DoomGamePython::setAutomapBufferEnabled)
        .def("set_automap_mode", &DoomGamePython::setAutomapMode)
        .def("set_automap_rotate", &DoomGamePython::setAutomapRotate)
        .def("set_automap_render_textures", &DoomGamePython::setAutomapRenderTextures)
        .def("is_objects_info_enabled", &DoomGamePython::isObjectsInfoEnabled)
        .def("set_objects_info_enabled", &DoomGamePython::setObjectsInfoEnabled)
        .def("is_sectors_info_enabled", &DoomGamePython::isSectorsInfoEnabled)
        .def("set_sectors_info_enabled", &DoomGamePython::setSectorsInfoEnabled)

        .def("set_render_hud", &DoomGamePython::setRenderHud)
        .def("set_render_minimal_hud", &DoomGamePython::setRenderMinimalHud)
        .def("set_render_weapon", &DoomGamePython::setRenderWeapon)
        .def("set_render_crosshair", &DoomGamePython::setRenderCrosshair)
        .def("set_render_decals", &DoomGamePython::setRenderDecals)
        .def("set_render_particles", &DoomGamePython::setRenderParticles)
        .def("set_render_effects_sprites", &DoomGamePython::setRenderEffectsSprites)
        .def("set_render_messages", &DoomGamePython::setRenderMessages)
        .def("set_render_corpses", &DoomGamePython::setRenderCorpses)
        .def("set_render_screen_flashes", &DoomGamePython::setRenderScreenFlashes)
        .def("set_render_all_frames", &DoomGamePython::setRenderAllFrames)
        .def("set_window_visible", &DoomGamePython::setWindowVisible)
        .def("get_screen_width", &DoomGamePython::getScreenWidth)
        .def("get_screen_height", &DoomGamePython::getScreenHeight)
        .def("get_screen_channels", &DoomGamePython::getScreenChannels)
        .def("get_screen_size", &DoomGamePython::getScreenSize)
        .def("get_screen_pitch", &DoomGamePython::getScreenPitch)
        .def("get_screen_format", &DoomGamePython::getScreenFormat);


    /* Utilities */
    /*----------------------------------------------------------------------------------------------------------------*/

    vz.def("doom_tics_to_ms", doomTicsToMs);
    vz.def("ms_to_doom_tics", msToDoomTics);
    vz.def("doom_tics_to_sec", doomTicsToSec);
    vz.def("sec_to_doom_tics", secToDoomTics);
    vz.def("doom_fixed_to_double", doomFixedToDouble_int);
    vz.def("doom_fixed_to_double", doomFixedToDouble_double);
    vz.def("doom_fixed_to_float", doomFixedToDouble_int);
    vz.def("doom_fixed_to_float", doomFixedToDouble_double);
    vz.def("is_binary_button", isBinaryButton);
    vz.def("is_delta_button", isDeltaButton);

}
