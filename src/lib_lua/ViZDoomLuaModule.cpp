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

#include "luabind/luabind.hpp"
#include "luabind/exception_handler.hpp"

using namespace vizdoom;
using namespace luabind;

/* C++ code to expose ViZDoom library via Lua */


/* Exceptions translation */
/*--------------------------------------------------------------------------------------------------------------------*/

#define EXCEPTION_TRANSLATE_TO_LUA(n) void translate ## n (lua_State* L, n const &e) { \
std::string error = std::string( #n ) + std::string(": ") + std::string(e.what()); \
lua_pushstring(L, error.c_str()); }
/*
 * void translateExceptionName(lua_State* L, exceptionName const &e) {
 * std::string error = std::string("exceptionName") + std::string(": ") + std::string(e.what());
 * lua_pushstring(L, error.c_str()); }
 */

EXCEPTION_TRANSLATE_TO_LUA(FileDoesNotExistException)
EXCEPTION_TRANSLATE_TO_LUA(MessageQueueException)
EXCEPTION_TRANSLATE_TO_LUA(SharedMemoryException)
EXCEPTION_TRANSLATE_TO_LUA(SignalException)
EXCEPTION_TRANSLATE_TO_LUA(ViZDoomIsNotRunningException)
EXCEPTION_TRANSLATE_TO_LUA(ViZDoomErrorException)
EXCEPTION_TRANSLATE_TO_LUA(ViZDoomUnexpectedExitException)


/* Overloaded functions */
/*--------------------------------------------------------------------------------------------------------------------*/

double (*doomFixedToDouble_int)(int) = &doomFixedToDouble;
double (*doomFixedToDouble_double)(double) = &doomFixedToDouble;


/* Module definition */
/*--------------------------------------------------------------------------------------------------------------------*/

extern "C" int luaopen_vizdoom(lua_State *luaState){

    open(luaState);


    /* Exceptions */
    /*----------------------------------------------------------------------------------------------------------------*/

    #define EXCEPTION_LUA_HANDLER(n) register_exception_handler< n >(&translate ## n);
    /*
     * register_exception_handler< ExceptionName >(&translateExceptionName);
     */

    EXCEPTION_LUA_HANDLER(FileDoesNotExistException)
    EXCEPTION_LUA_HANDLER(MessageQueueException)
    EXCEPTION_LUA_HANDLER(SharedMemoryException)
    EXCEPTION_LUA_HANDLER(SignalException)
    EXCEPTION_LUA_HANDLER(ViZDoomIsNotRunningException)
    EXCEPTION_LUA_HANDLER(ViZDoomErrorException)
    EXCEPTION_LUA_HANDLER(ViZDoomUnexpectedExitException)


    #define ENUM_VAL_2_LUA(v) value( #v , v )
    /* value("VALUE", VALUE) */

    #define ENUM_CLASS_VAL_2_LUA(c, v) value( #v , c::v )
    /* value("VALUE", (int)class::VALUE) */

    #define FUNC_2_LUA(f) def( #f , f )
    /* def("function", function) */

    #define CLASS_FUNC_2_LUA(c, f) .def( #f , &c::f )
    /* .def("function", &class::function) */
            

    module(luaState, "vizdoom")[

        /* Enums */
        /*------------------------------------------------------------------------------------------------------------*/

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

        class_<int>("AutomapMode")
            .enum_("AutomapMode")[
                ENUM_VAL_2_LUA(NORMAL),
                ENUM_VAL_2_LUA(WHOLE),
                ENUM_VAL_2_LUA(OBJECTS),
                ENUM_VAL_2_LUA(OBJECTS_WITH_SIZE)
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
                ENUM_VAL_2_LUA(MOVE_UP),
                ENUM_VAL_2_LUA(MOVE_DOWN),
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
                ENUM_VAL_2_LUA(POSITION_X),
                ENUM_VAL_2_LUA(POSITION_Y),
                ENUM_VAL_2_LUA(POSITION_Z),
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


        /* Structs */
        /*------------------------------------------------------------------------------------------------------------*/

        class_<GameStateLua>("GameState")
            .def_readonly("number", &GameStateLua::number)
            .def_readonly("gameVariables", &GameStateLua::gameVariables)
            .def_readonly("screenBuffer", &GameStateLua::screenBuffer)
            .def_readonly("depthBuffer", &GameStateLua::depthBuffer)
            .def_readonly("labelsBuffer", &GameStateLua::labelsBuffer)
            .def_readonly("automapBuffer", &GameStateLua::automapBuffer)
            .def_readonly("labels", &GameStateLua::labels),

        class_<Label>("Label")
            .def_readonly("objectId", &Label::objectId)
            .def_readonly("objectName", &Label::objectName)
            .def_readonly("value", &Label::value)
            .def_readonly("objectPositionX", &Label::objectPositionX)
            .def_readonly("objectPositionY", &Label::objectPositionY)
            .def_readonly("objectPositionZ", &Label::objectPositionZ),


        /* DoomGame */
        /*------------------------------------------------------------------------------------------------------------*/

        class_<DoomGameLua>("DoomGame")
            .def(constructor<>())
            CLASS_FUNC_2_LUA(DoomGameLua, init)
            CLASS_FUNC_2_LUA(DoomGameLua, loadConfig)
            CLASS_FUNC_2_LUA(DoomGameLua, close)
            .def("newEpisode", &DoomGameLua::newEpisode_)
            .def("newEpisode", &DoomGameLua::newEpisode_str)
            .def("replayEpisode", &DoomGameLua::replayEpisode_str)
            .def("replayEpisode", &DoomGameLua::replayEpisode_str_int)
            CLASS_FUNC_2_LUA(DoomGameLua, isEpisodeFinished)
            CLASS_FUNC_2_LUA(DoomGameLua, isNewEpisode)
            CLASS_FUNC_2_LUA(DoomGameLua, isPlayerDead)
            CLASS_FUNC_2_LUA(DoomGameLua, respawnPlayer)
            .def("setAction", &DoomGameLua::setAction)
            .def("_setAction", &DoomGameLua::setAction)
            .def("makeAction", &DoomGameLua::makeAction_obj)
            .def("_makeAction", &DoomGameLua::makeAction_obj_int)
            .def("advanceAction", &DoomGameLua::advanceAction_)
            .def("advanceAction", &DoomGameLua::advanceAction_int)
            .def("advanceAction", &DoomGameLua::advanceAction_int_bool)

            .def("getState", &DoomGameLua::getState)
            .def("_getState", &DoomGameLua::getState)

            CLASS_FUNC_2_LUA(DoomGameLua, getGameVariable)

            CLASS_FUNC_2_LUA(DoomGameLua, getLivingReward)
            CLASS_FUNC_2_LUA(DoomGameLua, setLivingReward)

            CLASS_FUNC_2_LUA(DoomGameLua, getDeathPenalty)
            CLASS_FUNC_2_LUA(DoomGameLua, setDeathPenalty)

            CLASS_FUNC_2_LUA(DoomGameLua, getLastReward)
            CLASS_FUNC_2_LUA(DoomGameLua, getTotalReward)

            .def("getLastAction", &DoomGameLua::getLastAction)
            .def("_getLastAction", &DoomGameLua::getLastAction)

            CLASS_FUNC_2_LUA(DoomGameLua, getAvailableGameVariables)
            CLASS_FUNC_2_LUA(DoomGameLua, setAvailableGameVariables)
            CLASS_FUNC_2_LUA(DoomGameLua, addAvailableGameVariable)
            CLASS_FUNC_2_LUA(DoomGameLua, clearAvailableGameVariables)
            CLASS_FUNC_2_LUA(DoomGameLua, getAvailableGameVariablesSize)

            CLASS_FUNC_2_LUA(DoomGameLua, getAvailableButtons)
            CLASS_FUNC_2_LUA(DoomGameLua, setAvailableButtons)
            .def("addAvailableButton", &DoomGameLua::addAvailableButton_btn)
            .def("addAvailableButton", &DoomGameLua::addAvailableButton_btn_int)
            CLASS_FUNC_2_LUA(DoomGameLua, clearAvailableButtons)
            CLASS_FUNC_2_LUA(DoomGameLua, getAvailableButtonsSize)
            CLASS_FUNC_2_LUA(DoomGameLua, setButtonMaxValue)
            CLASS_FUNC_2_LUA(DoomGameLua, getButtonMaxValue)

            CLASS_FUNC_2_LUA(DoomGameLua, addGameArgs)
            CLASS_FUNC_2_LUA(DoomGameLua, clearGameArgs)

            CLASS_FUNC_2_LUA(DoomGameLua, sendGameCommand)

            CLASS_FUNC_2_LUA(DoomGameLua, getMode)
            CLASS_FUNC_2_LUA(DoomGameLua, setMode)

            CLASS_FUNC_2_LUA(DoomGameLua, getTicrate)
            CLASS_FUNC_2_LUA(DoomGameLua, setTicrate)

            CLASS_FUNC_2_LUA(DoomGameLua, setViZDoomPath)
            CLASS_FUNC_2_LUA(DoomGameLua, setDoomGamePath)
            CLASS_FUNC_2_LUA(DoomGameLua, setDoomScenarioPath)
            CLASS_FUNC_2_LUA(DoomGameLua, setDoomMap)
            CLASS_FUNC_2_LUA(DoomGameLua, setDoomSkill)
            CLASS_FUNC_2_LUA(DoomGameLua, setDoomConfigPath)

            CLASS_FUNC_2_LUA(DoomGameLua, getSeed)
            CLASS_FUNC_2_LUA(DoomGameLua, setSeed)

            CLASS_FUNC_2_LUA(DoomGameLua, getEpisodeStartTime)
            CLASS_FUNC_2_LUA(DoomGameLua, setEpisodeStartTime)
            CLASS_FUNC_2_LUA(DoomGameLua, getEpisodeTimeout)
            CLASS_FUNC_2_LUA(DoomGameLua, setEpisodeTimeout)
            CLASS_FUNC_2_LUA(DoomGameLua, getEpisodeTime)

            CLASS_FUNC_2_LUA(DoomGameLua, setConsoleEnabled)
            CLASS_FUNC_2_LUA(DoomGameLua, setSoundEnabled)

            CLASS_FUNC_2_LUA(DoomGameLua, setScreenResolution)
            CLASS_FUNC_2_LUA(DoomGameLua, setScreenFormat)

            CLASS_FUNC_2_LUA(DoomGameLua, setDepthBufferEnabled)
            CLASS_FUNC_2_LUA(DoomGameLua, setLabelsBufferEnabled)
            CLASS_FUNC_2_LUA(DoomGameLua, setAutomapBufferEnabled)
            CLASS_FUNC_2_LUA(DoomGameLua, setAutomapMode)
            CLASS_FUNC_2_LUA(DoomGameLua, setAutomapRotate)
            CLASS_FUNC_2_LUA(DoomGameLua, setAutomapRenderTextures)

            CLASS_FUNC_2_LUA(DoomGameLua, setRenderHud)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderMinimalHud)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderWeapon)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderCrosshair)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderDecals)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderParticles)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderEffectsSprites)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderMessages)
            CLASS_FUNC_2_LUA(DoomGameLua, setRenderCorpses)
            CLASS_FUNC_2_LUA(DoomGameLua, getScreenWidth)
            CLASS_FUNC_2_LUA(DoomGameLua, getScreenHeight)
            CLASS_FUNC_2_LUA(DoomGameLua, getScreenChannels)
            CLASS_FUNC_2_LUA(DoomGameLua, getScreenSize)
            CLASS_FUNC_2_LUA(DoomGameLua, getScreenPitch)
            CLASS_FUNC_2_LUA(DoomGameLua, getScreenFormat)
            CLASS_FUNC_2_LUA(DoomGameLua, setWindowVisible),


        /* Utilities */
        /*------------------------------------------------------------------------------------------------------------*/

        FUNC_2_LUA(doomTicsToMs),
        FUNC_2_LUA(msToDoomTics),
        FUNC_2_LUA(doomTicsToSec),
        FUNC_2_LUA(secToDoomTics),
        //def("doomFixedToDouble", doomFixedToDouble_int),
        def("doomFixedToDouble", doomFixedToDouble_double),
        //def("doomFixedToNumber", doomFixedToDouble_int),
        def("doomFixedToNumber", doomFixedToDouble_double),
        FUNC_2_LUA(isBinaryButton),
        FUNC_2_LUA(isDeltaButton)

    ];

    return 1;
}

extern "C" int luaopen_vizdoom_vizdoom(lua_State *luaState) {
    return luaopen_vizdoom(luaState);
}