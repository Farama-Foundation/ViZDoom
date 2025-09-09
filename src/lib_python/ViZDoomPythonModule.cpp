/*
 Copyright (C) 2016 by Wojciech Jaśkowski, Michał Kempka, Grzegorz Runc, Jakub Toczek, Marek Wydmuch
 Copyright (C) 2017 - 2022 by Marek Wydmuch, Michał Kempka, Wojciech Jaśkowski, and the respective contributors
 Copyright (C) 2023 - 2024 by Marek Wydmuch, Farama Foundation, and the respective contributors

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
#include "ViZDoomMethodsDocstrings.h"
#include "ViZDoomObjectsDocstrings.h"

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

    vz.doc() = "ViZDoom Python module.";
    vz.attr("__version__") = pyb::str(VIZDOOM_LIB_VERSION_STR);

    /* Exceptions */
    /*----------------------------------------------------------------------------------------------------------------*/

    #define EXCEPTION_TO_PYT(n) pyb::register_exception< n >(vz , #n);
    /* register_exception< ExceptionName >(vz, "ExceptionName"); */

    EXCEPTION_TO_PYT(FileDoesNotExistException)
    EXCEPTION_TO_PYT(MessageQueueException)
    EXCEPTION_TO_PYT(SharedMemoryException)
    EXCEPTION_TO_PYT(ViZDoomIsNotRunningException)
    EXCEPTION_TO_PYT(ViZDoomErrorException)
    EXCEPTION_TO_PYT(ViZDoomUnexpectedExitException)
    EXCEPTION_TO_PYT(ViZDoomNoOpenALSoundException)

    /* Helpers */
    /*----------------------------------------------------------------------------------------------------------------*/
    // These macros are used to make the code below shorter and less prone to typos

    #define CONST_2_PYT(c) vz.attr( #c ) = c
    /* vz.attr("CONST") = CONST */

    #define ENUM_VAL_2_PYT(v) .value( #v , v )
    /* .value("VALUE", VALUE) */

    #define ENUM_CLASS_VAL_2_PYT(c, v) .value( #v , c::v )
    /* .value("VALUE", class::VALUE) */

    #define FUNC_2_PYT(n, f) vz.def( n , f , pyb::doc(docstrings::f) )
    /* vz.def("name", function, docstrings::function) */

    #define FUNC_2_PYT_WITH_ARGS(n, f, ...) vz.def( n , f , pyb::doc(docstrings::f) , __VA_ARGS__ )
    /* vz.def("name", function, docstrings::function, args) */

    #define CLASS_FUNC_2_PYT(n, cf) .def( n , &cf , pyb::doc(docstrings::cf) )
    /* .def("name", &class::function, docstrings::class::function) */

    #define CLASS_FUNC_2_PYT_WITH_ARGS(n, cf, ...) .def( n , &cf , pyb::doc(docstrings::cf) , __VA_ARGS__ )
    /* .def("name", &class::function, docstrings::class::function, args) */


    /* Consts */
    /*----------------------------------------------------------------------------------------------------------------*/

    CONST_2_PYT(SLOT_COUNT);
    CONST_2_PYT(MAX_PLAYERS);
    CONST_2_PYT(MAX_PLAYER_NAME_LENGTH);
    CONST_2_PYT(USER_VARIABLE_COUNT);
    CONST_2_PYT(DEFAULT_TICRATE);
    CONST_2_PYT(DEFAULT_FPS);
    CONST_2_PYT(DEFAULT_FRAMETIME_MS);
    CONST_2_PYT(DEFAULT_FRAMETIME_S);

    CONST_2_PYT(BINARY_BUTTON_COUNT);
    CONST_2_PYT(DELTA_BUTTON_COUNT);
    CONST_2_PYT(BUTTON_COUNT);


    /* Enums */
    /*----------------------------------------------------------------------------------------------------------------*/

    pyb::enum_<Mode>(vz, "Mode", docstrings::Mode)
        ENUM_VAL_2_PYT(PLAYER)
        ENUM_VAL_2_PYT(SPECTATOR)
        ENUM_VAL_2_PYT(ASYNC_PLAYER)
        ENUM_VAL_2_PYT(ASYNC_SPECTATOR)
        .export_values();

    pyb::enum_<ScreenFormat>(vz, "ScreenFormat", docstrings::ScreenFormat)
        ENUM_VAL_2_PYT(CRCGCB)
        ENUM_VAL_2_PYT(RGB24)
        ENUM_VAL_2_PYT(RGBA32)
        ENUM_VAL_2_PYT(ARGB32)
        ENUM_VAL_2_PYT(CBCGCR)
        ENUM_VAL_2_PYT(BGR24)
        ENUM_VAL_2_PYT(BGRA32)
        ENUM_VAL_2_PYT(ABGR32)
        ENUM_VAL_2_PYT(GRAY8)
        ENUM_VAL_2_PYT(DOOM_256_COLORS8)
        .export_values();

    pyb::enum_<ScreenResolution>(vz, "ScreenResolution", docstrings::ScreenResolution)
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

    pyb::enum_<AutomapMode>(vz, "AutomapMode", docstrings::AutomapMode)
        ENUM_VAL_2_PYT(NORMAL)
        ENUM_VAL_2_PYT(WHOLE)
        ENUM_VAL_2_PYT(OBJECTS)
        ENUM_VAL_2_PYT(OBJECTS_WITH_SIZE)
        .export_values();

    pyb::enum_<Button>(vz, "Button", docstrings::Button)
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

    pyb::enum_<GameVariable>(vz, "GameVariable", docstrings::GameVariable)
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

    pyb::enum_<SamplingRate>(vz, "SamplingRate", docstrings::SamplingRate)
        ENUM_VAL_2_PYT(SR_11025)
        ENUM_VAL_2_PYT(SR_22050)
        ENUM_VAL_2_PYT(SR_44100)
        .export_values();


    /* Structs */
    /*----------------------------------------------------------------------------------------------------------------*/

    pyb::class_<Label>(vz, "Label", docstrings::Label)
        .def(pyb::pickle(
            [](const Label& o) { // dump
                return pyb::make_tuple(
                    o.value, 
                    o.x, 
                    o.y, 
                    o.width, 
                    o.height,
                    o.objectId,
                    o.objectPositionX, 
                    o.objectPositionY, 
                    o.objectPositionZ,
                    o.objectAngle,
                    o.objectPitch,
                    o.objectRoll,
                    o.objectVelocityX,
                    o.objectVelocityY,
                    o.objectVelocityZ,
                    o.objectName,
                    o.objectCategory
                );
            },
            [](pyb::tuple t) { // load
                return Label{
                    t[0].cast<uint8_t>(), 
                    t[1].cast<unsigned int>(),
                    t[2].cast<unsigned int>(), 
                    t[3].cast<unsigned int>(), 
                    t[4].cast<unsigned int>(), 
                    t[5].cast<unsigned int>(),
                    t[6].cast<double>(),
                    t[7].cast<double>(),
                    t[8].cast<double>(),
                    t[9].cast<double>(),
                    t[10].cast<double>(),
                    t[11].cast<double>(),
                    t[12].cast<double>(),
                    t[13].cast<double>(),
                    t[14].cast<double>(),
                    t[15].cast<std::string>(),
                    t[16].cast<std::string>()
                };
            })
        )
        .def_readonly("value", &Label::value)
        .def_readonly("x", &Label::x)
        .def_readonly("y", &Label::y)
        .def_readonly("width", &Label::width)
        .def_readonly("height", &Label::height)
        .def_readonly("object_id", &Label::objectId)
        .def_readonly("object_position_x", &Label::objectPositionX)
        .def_readonly("object_position_y", &Label::objectPositionY)
        .def_readonly("object_position_z", &Label::objectPositionZ)
        .def_readonly("object_angle", &Label::objectAngle)
        .def_readonly("object_pitch", &Label::objectPitch)
        .def_readonly("object_roll", &Label::objectRoll)
        .def_readonly("object_velocity_x", &Label::objectVelocityX)
        .def_readonly("object_velocity_y", &Label::objectVelocityY)
        .def_readonly("object_velocity_z", &Label::objectVelocityZ)
        .def_readonly("object_name", &Label::objectName)
        .def_readonly("object_category", &Label::objectCategory);

    pyb::class_<Object>(vz, "Object", docstrings::Object)
            .def(pyb::pickle(
            [](const Object& o) { // dump
                return pyb::make_tuple(
                    o.id,
                    o.positionX, 
                    o.positionY, 
                    o.positionZ,
                    o.angle,
                    o.pitch,
                    o.roll,
                    o.velocityX,
                    o.velocityY,
                    o.velocityZ,
                    o.name
                );
            },
            [](pyb::tuple t) { // load
                return Object{
                    t[0].cast<unsigned int>(), 
                    t[1].cast<double>(),
                    t[2].cast<double>(),
                    t[3].cast<double>(),
                    t[4].cast<double>(),
                    t[5].cast<double>(),
                    t[6].cast<double>(),
                    t[7].cast<double>(),
                    t[8].cast<double>(),
                    t[9].cast<double>(),
                    t[10].cast<std::string>()
                };
            })
        )
        .def_readonly("id", &Object::id)
        .def_readonly("position_x", &Object::positionX)
        .def_readonly("position_y", &Object::positionY)
        .def_readonly("position_z", &Object::positionZ)
        .def_readonly("angle", &Object::angle)
        .def_readonly("pitch", &Object::pitch)
        .def_readonly("roll", &Object::roll)
        .def_readonly("velocity_x", &Object::velocityX)
        .def_readonly("velocity_y", &Object::velocityY)
        .def_readonly("velocity_z", &Object::velocityZ)
        .def_readonly("name", &Object::name);

    pyb::class_<Line>(vz, "Line", docstrings::Line)
        .def(pyb::pickle(
            [](const Line& o) { // dump
                return pyb::make_tuple(
                    o.x1, 
                    o.y1, 
                    o.x2,
                    o.y2,
                    o.isBlocking
                );
            },
            [](pyb::tuple t) { // load
                return Line{
                    t[0].cast<double>(), 
                    t[1].cast<double>(),
                    t[2].cast<double>(),
                    t[3].cast<double>(),
                    t[4].cast<bool>()
                };
            })
        )
        .def_readonly("x1", &Line::x1)
        .def_readonly("y1", &Line::y1)
        .def_readonly("x2", &Line::x2)
        .def_readonly("y2", &Line::y2)
        .def_readonly("is_blocking", &Line::isBlocking);

    pyb::class_<SectorPython>(vz, "Sector", docstrings::Sector)
        .def(pyb::pickle(
            [](const SectorPython& o) { // dump
                return pyb::make_tuple(
                    o.floorHeight, 
                    o.ceilingHeight, 
                    o.lines
                );
            },
            [](pyb::tuple t) { // load
                return SectorPython{
                    t[0].cast<double>(), 
                    t[1].cast<double>(),
                    t[2].cast<pyb::list>()
                };
            })
        )
        .def_readonly("floor_height", &SectorPython::floorHeight)
        .def_readonly("ceiling_height", &SectorPython::ceilingHeight)
        .def_readonly("lines", &SectorPython::lines);

    pyb::class_<GameStatePython>(vz, "GameState", docstrings::GameState)
        .def(pyb::pickle(
            [](const GameStatePython& o) { // dump
                return pyb::make_tuple(
                    o.number, 
                    o.tic, 
                    o.gameVariables, 
                    o.screenBuffer, 
                    o.depthBuffer,
                    o.labelsBuffer,
                    o.automapBuffer, 
                    o.audioBuffer, 
                    o.labels,
                    o.objects,
                    o.sectors,
                    o.notificationsBuffer
                );
            },
            [](pyb::tuple t) { // load
                return GameStatePython{
                    t[0].cast<unsigned int>(), 
                    t[1].cast<unsigned int>(),
                    t[2].cast<pyb::object>(), 
                    t[3].cast<pyb::object>(), 
                    t[4].cast<pyb::object>(), 
                    t[5].cast<pyb::object>(),
                    t[6].cast<pyb::object>(),
                    t[7].cast<pyb::object>(),
                    t[8].cast<pyb::object>(),
                    t[9].cast<pyb::object>(),
                    t[10].cast<pyb::object>(),
                    t[11].cast<pyb::object>()
                };
            })
        )
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
        .def_readonly("sectors", &GameStatePython::sectors)

        .def_readonly("notifications_buffer", &GameStatePython::notificationsBuffer)
        ;

    pyb::class_<ServerStatePython>(vz, "ServerState", docstrings::ServerState)
            .def(pyb::pickle(
            [](const ServerStatePython& o) { // dump
                return pyb::make_tuple(
                    o.tic, 
                    o.playerCount, 
                    o.playersInGame, 
                    o.playersNames,
                    o.playersFrags,
                    o.playersAfk, 
                    o.playersLastActionTic, 
                    o.playersLastKillTic
                );
            },
            [](pyb::tuple t) { // load
                return ServerStatePython{
                    t[0].cast<unsigned int>(), 
                    t[1].cast<unsigned int>(),
                    t[2].cast<pyb::list>(), 
                    t[3].cast<pyb::list>(), 
                    t[4].cast<pyb::list>(), 
                    t[5].cast<pyb::list>(),
                    t[6].cast<pyb::list>(),
                    t[7].cast<pyb::list>()
                };
            })
        )
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

    pyb::class_<DoomGamePython>(vz, "DoomGame", docstrings::DoomGame)
        .def(pyb::init<>())
        CLASS_FUNC_2_PYT("init", DoomGamePython::init)
        CLASS_FUNC_2_PYT_WITH_ARGS("load_config", DoomGamePython::loadConfig, pyb::arg("config"))
        CLASS_FUNC_2_PYT("close", DoomGamePython::close)
        CLASS_FUNC_2_PYT("is_running", DoomGamePython::isRunning)
        CLASS_FUNC_2_PYT("is_multiplayer_game", DoomGamePython::isMultiplayerGame)
        CLASS_FUNC_2_PYT("is_recording_episode", DoomGamePython::isRecordingEpisode)
        CLASS_FUNC_2_PYT("is_replaying_episode", DoomGamePython::isReplayingEpisode)
        CLASS_FUNC_2_PYT_WITH_ARGS("new_episode", DoomGamePython::newEpisode, pyb::arg("recording_file_path") = "")
        CLASS_FUNC_2_PYT_WITH_ARGS("replay_episode", DoomGamePython::replayEpisode, pyb::arg("file_path"), pyb::arg("player") = 0)
        CLASS_FUNC_2_PYT("is_episode_finished", DoomGamePython::isEpisodeFinished)
        CLASS_FUNC_2_PYT("is_episode_timeout_reached", DoomGamePython::isEpisodeTimeoutReached)
        CLASS_FUNC_2_PYT("is_new_episode", DoomGamePython::isNewEpisode)
        CLASS_FUNC_2_PYT("is_player_dead", DoomGamePython::isPlayerDead)
        CLASS_FUNC_2_PYT("respawn_player", DoomGamePython::respawnPlayer)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_action", DoomGamePython::setAction, pyb::arg("action"))
        CLASS_FUNC_2_PYT_WITH_ARGS("make_action", DoomGamePython::makeAction, pyb::arg("action"), pyb::arg("tics") = 1)
        CLASS_FUNC_2_PYT_WITH_ARGS("advance_action", DoomGamePython::advanceAction, pyb::arg("tics") = 1, pyb::arg("update_state") = true)
        CLASS_FUNC_2_PYT_WITH_ARGS("save", DoomGamePython::save, pyb::arg("file_path"))
        CLASS_FUNC_2_PYT_WITH_ARGS("load", DoomGamePython::load, pyb::arg("file_path"))

        .def("get_state", &DoomGamePython::getState, pyb::return_value_policy::take_ownership, docstrings::DoomGamePython::getState)
        .def("get_server_state", &DoomGamePython::getServerState, pyb::return_value_policy::take_ownership, docstrings::DoomGamePython::getServerState)

        CLASS_FUNC_2_PYT_WITH_ARGS("get_game_variable", DoomGamePython::getGameVariable, pyb::arg("variable"))
        CLASS_FUNC_2_PYT_WITH_ARGS("get_button", DoomGamePython::getButton, pyb::arg("button"))

        CLASS_FUNC_2_PYT("get_living_reward", DoomGamePython::getLivingReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_living_reward", DoomGamePython::setLivingReward, pyb::arg("living_reward"))

        CLASS_FUNC_2_PYT("get_death_penalty", DoomGamePython::getDeathPenalty)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_death_penalty", DoomGamePython::setDeathPenalty, pyb::arg("death_penalty"))
        CLASS_FUNC_2_PYT("get_death_reward", DoomGamePython::getDeathReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_death_reward", DoomGamePython::setDeathReward, pyb::arg("death_reward"))
        CLASS_FUNC_2_PYT("get_map_exit_reward", DoomGamePython::getMapExitReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_map_exit_reward", DoomGamePython::setMapExitReward, pyb::arg("map_exit_reward"))
        
        CLASS_FUNC_2_PYT("get_kill_reward", DoomGamePython::getKillReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_kill_reward", DoomGamePython::setKillReward, pyb::arg("kill_reward"))
        CLASS_FUNC_2_PYT("get_secret_reward", DoomGamePython::getSecretReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_secret_reward", DoomGamePython::setSecretReward, pyb::arg("secret_reward"))
        CLASS_FUNC_2_PYT("get_item_reward", DoomGamePython::getItemReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_item_reward", DoomGamePython::setItemReward, pyb::arg("item_reward"))
        CLASS_FUNC_2_PYT("get_frag_reward", DoomGamePython::getFragReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_frag_reward", DoomGamePython::setFragReward, pyb::arg("frag_reward"))
        CLASS_FUNC_2_PYT("get_hit_reward", DoomGamePython::getHitReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_hit_reward", DoomGamePython::setHitReward, pyb::arg("hit_reward"))
        CLASS_FUNC_2_PYT("get_hit_taken_reward", DoomGamePython::getHitTakenReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_hit_taken_reward", DoomGamePython::setHitTakenReward, pyb::arg("hit_taken_reward"))
        CLASS_FUNC_2_PYT("get_hit_taken_penalty", DoomGamePython::getHitTakenPenalty)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_hit_taken_penalty", DoomGamePython::setHitTakenPenalty, pyb::arg("hit_taken_penalty"))
        CLASS_FUNC_2_PYT("get_damage_made_reward", DoomGamePython::getDamageMadeReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_damage_made_reward", DoomGamePython::setDamageMadeReward, pyb::arg("damage_made_reward"))
        CLASS_FUNC_2_PYT("get_damage_taken_reward", DoomGamePython::getDamageTakenReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_damage_taken_reward", DoomGamePython::setDamageTakenReward, pyb::arg("damage_taken_reward"))
        CLASS_FUNC_2_PYT("get_damage_taken_penalty", DoomGamePython::getDamageTakenPenalty)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_damage_taken_penalty", DoomGamePython::setDamageTakenPenalty, pyb::arg("damage_taken_penalty"))
        CLASS_FUNC_2_PYT("get_health_reward", DoomGamePython::getHealthReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_health_reward", DoomGamePython::setHealthReward, pyb::arg("health_reward"))
        CLASS_FUNC_2_PYT("get_armor_reward", DoomGamePython::getArmorReward)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_armor_reward", DoomGamePython::setArmorReward, pyb::arg("armor_reward"))

        CLASS_FUNC_2_PYT("get_last_reward", DoomGamePython::getLastReward)
        CLASS_FUNC_2_PYT("get_total_reward", DoomGamePython::getTotalReward)

        CLASS_FUNC_2_PYT("get_last_action", DoomGamePython::getLastAction)

        CLASS_FUNC_2_PYT("get_available_game_variables", DoomGamePython::getAvailableGameVariables)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_available_game_variables", DoomGamePython::setAvailableGameVariables, pyb::arg("variables"))
        CLASS_FUNC_2_PYT_WITH_ARGS("add_available_game_variable", DoomGamePython::addAvailableGameVariable, pyb::arg("variable"))
        CLASS_FUNC_2_PYT("clear_available_game_variables", DoomGamePython::clearAvailableGameVariables)
        CLASS_FUNC_2_PYT("get_available_game_variables_size", DoomGamePython::getAvailableGameVariablesSize)

        CLASS_FUNC_2_PYT("get_available_buttons", DoomGamePython::getAvailableButtons)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_available_buttons", DoomGamePython::setAvailableButtons, pyb::arg("buttons"))
        CLASS_FUNC_2_PYT_WITH_ARGS("add_available_button", DoomGamePython::addAvailableButton, pyb::arg("button"), pyb::arg("max_value") = -1)
        CLASS_FUNC_2_PYT("clear_available_buttons", DoomGamePython::clearAvailableButtons)
        CLASS_FUNC_2_PYT("get_available_buttons_size", DoomGamePython::getAvailableButtonsSize)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_button_max_value", DoomGamePython::setButtonMaxValue, pyb::arg("button"), pyb::arg("max_value"))
        CLASS_FUNC_2_PYT_WITH_ARGS("get_button_max_value", DoomGamePython::getButtonMaxValue, pyb::arg("button"))

        CLASS_FUNC_2_PYT_WITH_ARGS("set_game_args", DoomGamePython::setGameArgs, pyb::arg("args"))
        CLASS_FUNC_2_PYT_WITH_ARGS("add_game_args", DoomGamePython::addGameArgs, pyb::arg("args"))
        CLASS_FUNC_2_PYT("clear_game_args", DoomGamePython::clearGameArgs)
        CLASS_FUNC_2_PYT("get_game_args", DoomGamePython::getGameArgs)

        CLASS_FUNC_2_PYT_WITH_ARGS("send_game_command", DoomGamePython::sendGameCommand, pyb::arg("cmd"))

        CLASS_FUNC_2_PYT("get_mode", DoomGamePython::getMode)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_mode", DoomGamePython::setMode, pyb::arg("mode"))

        CLASS_FUNC_2_PYT("get_ticrate", DoomGamePython::getTicrate)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_ticrate", DoomGamePython::setTicrate, pyb::arg("button"))

        CLASS_FUNC_2_PYT_WITH_ARGS("set_vizdoom_path", DoomGamePython::setViZDoomPath, pyb::arg("button"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_doom_game_path", DoomGamePython::setDoomGamePath, pyb::arg("button"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_doom_scenario_path", DoomGamePython::setDoomScenarioPath, pyb::arg("button"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_doom_map", DoomGamePython::setDoomMap, pyb::arg("button"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_doom_skill", DoomGamePython::setDoomSkill, pyb::arg("button"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_doom_config_path", DoomGamePython::setDoomConfigPath, pyb::arg("button"))

        CLASS_FUNC_2_PYT("get_seed", DoomGamePython::getSeed)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_seed", DoomGamePython::setSeed, pyb::arg("seed"))

        CLASS_FUNC_2_PYT("get_episode_start_time", DoomGamePython::getEpisodeStartTime)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_episode_start_time", DoomGamePython::setEpisodeStartTime, pyb::arg("start_time"))
        CLASS_FUNC_2_PYT("get_episode_timeout", DoomGamePython::getEpisodeTimeout)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_episode_timeout", DoomGamePython::setEpisodeTimeout, pyb::arg("timeout"))
        CLASS_FUNC_2_PYT("get_episode_time", DoomGamePython::getEpisodeTime)

        CLASS_FUNC_2_PYT_WITH_ARGS("set_console_enabled", DoomGamePython::setConsoleEnabled, pyb::arg("console"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_sound_enabled", DoomGamePython::setSoundEnabled, pyb::arg("sound"))

        CLASS_FUNC_2_PYT("is_audio_buffer_enabled", DoomGamePython::isAudioBufferEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_audio_buffer_enabled", DoomGamePython::setAudioBufferEnabled, pyb::arg("audio_buffer"))
        CLASS_FUNC_2_PYT("get_audio_sampling_rate", DoomGamePython::getAudioSamplingRate)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_audio_sampling_rate", DoomGamePython::setAudioSamplingRate, pyb::arg("sampling_rate"))
        CLASS_FUNC_2_PYT("get_audio_buffer_size", DoomGamePython::getAudioBufferSize)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_audio_buffer_size", DoomGamePython::setAudioBufferSize, pyb::arg("tics"))
        
        CLASS_FUNC_2_PYT("is_notifications_buffer_enabled", DoomGamePython::isNotificationsBufferEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_notifications_buffer_enabled", DoomGamePython::setNotificationsBufferEnabled, pyb::arg("notifications_buffer"))
        CLASS_FUNC_2_PYT("get_notifications_buffer_size", DoomGamePython::getNotificationsBufferSize)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_notifications_buffer_size", DoomGamePython::setNotificationsBufferSize, pyb::arg("tics"))

        CLASS_FUNC_2_PYT_WITH_ARGS("set_screen_resolution", DoomGamePython::setScreenResolution, pyb::arg("resolution"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_screen_format", DoomGamePython::setScreenFormat, pyb::arg("format"))

        CLASS_FUNC_2_PYT("is_depth_buffer_enabled", DoomGamePython::isDepthBufferEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_depth_buffer_enabled", DoomGamePython::setDepthBufferEnabled, pyb::arg("depth_buffer"))
        CLASS_FUNC_2_PYT("is_labels_buffer_enabled", DoomGamePython::isLabelsBufferEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_labels_buffer_enabled", DoomGamePython::setLabelsBufferEnabled, pyb::arg("labels_buffer"))
        CLASS_FUNC_2_PYT("is_automap_buffer_enabled", DoomGamePython::isAutomapBufferEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_automap_buffer_enabled", DoomGamePython::setAutomapBufferEnabled, pyb::arg("automap_buffer"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_automap_mode", DoomGamePython::setAutomapMode, pyb::arg("mode"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_automap_rotate", DoomGamePython::setAutomapRotate, pyb::arg("rotate"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_automap_render_textures", DoomGamePython::setAutomapRenderTextures, pyb::arg("textures"))
        CLASS_FUNC_2_PYT("is_objects_info_enabled", DoomGamePython::isObjectsInfoEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_objects_info_enabled", DoomGamePython::setObjectsInfoEnabled, pyb::arg("objects_info"))
        CLASS_FUNC_2_PYT("is_sectors_info_enabled", DoomGamePython::isSectorsInfoEnabled)
        CLASS_FUNC_2_PYT_WITH_ARGS("set_sectors_info_enabled", DoomGamePython::setSectorsInfoEnabled, pyb::arg("sectors_info"))

        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_hud", DoomGamePython::setRenderHud, pyb::arg("hud"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_minimal_hud", DoomGamePython::setRenderMinimalHud, pyb::arg("min_hud"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_weapon", DoomGamePython::setRenderWeapon, pyb::arg("weapon"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_crosshair", DoomGamePython::setRenderCrosshair, pyb::arg("crosshair"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_decals", DoomGamePython::setRenderDecals, pyb::arg("decals"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_particles", DoomGamePython::setRenderParticles, pyb::arg("particles"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_effects_sprites", DoomGamePython::setRenderEffectsSprites, pyb::arg("sprites"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_messages", DoomGamePython::setRenderMessages, pyb::arg("messages"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_corpses", DoomGamePython::setRenderCorpses, pyb::arg("bodies"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_screen_flashes", DoomGamePython::setRenderScreenFlashes, pyb::arg("flashes"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_render_all_frames", DoomGamePython::setRenderAllFrames, pyb::arg("all_frames"))
        CLASS_FUNC_2_PYT_WITH_ARGS("set_window_visible", DoomGamePython::setWindowVisible, pyb::arg("visiblity"))
        CLASS_FUNC_2_PYT("get_screen_width", DoomGamePython::getScreenWidth)
        CLASS_FUNC_2_PYT("get_screen_height", DoomGamePython::getScreenHeight)
        CLASS_FUNC_2_PYT("get_screen_channels", DoomGamePython::getScreenChannels)
        CLASS_FUNC_2_PYT("get_screen_size", DoomGamePython::getScreenSize)
        CLASS_FUNC_2_PYT("get_screen_pitch", DoomGamePython::getScreenPitch)
        CLASS_FUNC_2_PYT("get_screen_format", DoomGamePython::getScreenFormat);


    /* Utilities */
    /*----------------------------------------------------------------------------------------------------------------*/

    FUNC_2_PYT_WITH_ARGS("doom_tics_to_ms", doomTicsToMs, pyb::arg("doom_tics"), pyb::arg("fps") = 35);
    FUNC_2_PYT_WITH_ARGS("ms_to_doom_tics", msToDoomTics, pyb::arg("doom_tics"), pyb::arg("fps") = 35);
    FUNC_2_PYT_WITH_ARGS("doom_tics_to_sec", doomTicsToSec, pyb::arg("doom_tics"), pyb::arg("fps") = 35);
    FUNC_2_PYT_WITH_ARGS("sec_to_doom_tics", secToDoomTics, pyb::arg("doom_tics"), pyb::arg("fps") = 35);
    vz.def("doom_fixed_to_double", doomFixedToDouble_int, docstrings::doomFixedToDouble, pyb::arg("doom_fixed"));
    vz.def("doom_fixed_to_double", doomFixedToDouble_double, docstrings::doomFixedToDouble, pyb::arg("doom_fixed"));
    vz.def("doom_fixed_to_float", doomFixedToDouble_int, docstrings::doomFixedToDouble, pyb::arg("doom_fixed"));
    vz.def("doom_fixed_to_float", doomFixedToDouble_double, docstrings::doomFixedToDouble, pyb::arg("doom_fixed"));
    FUNC_2_PYT_WITH_ARGS("is_binary_button", isBinaryButton, pyb::arg("button"));
    FUNC_2_PYT_WITH_ARGS("is_delta_button", isDeltaButton, pyb::arg("button"));
    vz.def("get_default_categories", getDefaultCategories, pyb::doc(docstrings::getDefaultCategories));

}
