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

#ifndef __VIZDOOM_TYPES_H__
#define __VIZDOOM_TYPES_H__

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace vizdoom{

    typedef std::vector<uint8_t> Buffer;
    typedef std::shared_ptr<Buffer> BufferPtr;

    struct Label{
        unsigned int objectId;
        std::string objectName;
        uint8_t value;
        double objectPositionX;
        double objectPositionY;
        double objectPositionZ;
    };

    struct GameState {
        unsigned int number;

        std::vector<double> gameVariables;

        BufferPtr screenBuffer;
        BufferPtr depthBuffer;
        BufferPtr labelsBuffer;
        BufferPtr automapBuffer;

        std::vector<Label> labels;
    };

    typedef std::shared_ptr<GameState> GameStatePtr;

    enum Mode {
        PLAYER,             // synchronous player mode
        SPECTATOR,          // synchronous spectator mode
        ASYNC_PLAYER,       // asynchronous player mode
        ASYNC_SPECTATOR,    // asynchronous spectator mode
    };

    enum ScreenFormat {
        CRCGCB              = 0, // 3 channels of 8-bit values in RGB order
        RGB24               = 1, // channel of RGB values stored in 24 bits, where R value is stored in the oldest 8 bits
        RGBA32              = 2, // channel of RGBA values stored in 32 bits, where R value is stored in the oldest 8 bits
        ARGB32              = 3, // channel of ARGB values stored in 32 bits, where A value is stored in the oldest 8 bits
        CBCGCR              = 4, // 3 channels of 8-bit values in BGR order
        BGR24               = 5, // channel of BGR values stored in 24 bits, where B value is stored in the oldest 8 bits
        BGRA32              = 6, // channel of BGRA values stored in 32 bits, where B value is stored in the oldest 8 bits
        ABGR32              = 7, // channel of ABGR values stored in 32 bits, where A value is stored in the oldest 8 bits
        GRAY8               = 8, // 8-bit gray channel
        DOOM_256_COLORS8    = 9, // 8-bit channel with Doom palette values
    };

    enum ScreenResolution {
        RES_160X120,    // 4:3

        RES_200X125,    // 16:10
        RES_200X150,    // 4:3

        RES_256X144,    // 16:9
        RES_256X160,    // 16:10
        RES_256X192,    // 4:3

        RES_320X180,    // 16:9
        RES_320X200,    // 16:10
        RES_320X240,    // 4:3
        RES_320X256,    // 5:4

        RES_400X225,    // 16:9
        RES_400X250,    // 16:10
        RES_400X300,    // 4:3

        RES_512X288,    // 16:9
        RES_512X320,    // 16:10
        RES_512X384,    // 4:3

        RES_640X360,    // 16:9
        RES_640X400,    // 16:10
        RES_640X480,    // 4:3

        RES_800X450,    // 16:9
        RES_800X500,    // 16:10
        RES_800X600,    // 4:3

        RES_1024X576,   // 16:9
        RES_1024X640,   // 16:10
        RES_1024X768,   // 4:3

        RES_1280X720,   // 16:9
        RES_1280X800,   // 16:10
        RES_1280X960,   // 4:3
        RES_1280X1024,  // 5:4

        RES_1400X787,   // 16:9
        RES_1400X875,   // 16:10
        RES_1400X1050,  // 4:3

        RES_1600X900,   // 16:9
        RES_1600X1000,  // 16:10
        RES_1600X1200,  // 4:3

        RES_1920X1080,  // 16:9
    };

    enum AutomapMode {
        NORMAL,             // Only level architecture the player has seen is shown.
        WHOLE,              // All architecture is shown, regardless of whether or not the player has seen it.
        OBJECTS,            // In addition to the previous, shows all things in the map as arrows pointing in
                            // the direction they are facing.
        OBJECTS_WITH_SIZE,  // In addition to the previous, all things are wrapped in a box showing their size.
    };

    enum GameVariable {

        /* Defined variables */
        KILLCOUNT,
        ITEMCOUNT,
        SECRETCOUNT,
        FRAGCOUNT,
        DEATHCOUNT,
        HEALTH,
        ARMOR,
        DEAD,
        ON_GROUND,
        ATTACK_READY,
        ALTATTACK_READY,
        SELECTED_WEAPON,
        SELECTED_WEAPON_AMMO,
        AMMO0,
        AMMO1,
        AMMO2,
        AMMO3,
        AMMO4,
        AMMO5,
        AMMO6,
        AMMO7,
        AMMO8,
        AMMO9,
        WEAPON0,
        WEAPON1,
        WEAPON2,
        WEAPON3,
        WEAPON4,
        WEAPON5,
        WEAPON6,
        WEAPON7,
        WEAPON8,
        WEAPON9,

        POSITION_X,
        POSITION_Y,
        POSITION_Z,

        PLAYER_NUMBER,
        PLAYER_COUNT,
        PLAYER1_FRAGCOUNT,
        PLAYER2_FRAGCOUNT,
        PLAYER3_FRAGCOUNT,
        PLAYER4_FRAGCOUNT,
        PLAYER5_FRAGCOUNT,
        PLAYER6_FRAGCOUNT,
        PLAYER7_FRAGCOUNT,
        PLAYER8_FRAGCOUNT,

        /* User (ACS) variables */
        // USER0 is reserved for reward
        USER1,
        USER2,
        USER3,
        USER4,
        USER5,
        USER6,
        USER7,
        USER8,
        USER9,
        USER10,
        USER11,
        USER12,
        USER13,
        USER14,
        USER15,
        USER16,
        USER17,
        USER18,
        USER19,
        USER20,
        USER21,
        USER22,
        USER23,
        USER24,
        USER25,
        USER26,
        USER27,
        USER28,
        USER29,
        USER30,
        USER31,
        USER32,
        USER33,
        USER34,
        USER35,
        USER36,
        USER37,
        USER38,
        USER39,
        USER40,
        USER41,
        USER42,
        USER43,
        USER44,
        USER45,
        USER46,
        USER47,
        USER48,
        USER49,
        USER50,
        USER51,
        USER52,
        USER53,
        USER54,
        USER55,
        USER56,
        USER57,
        USER58,
        USER59,
        USER60,
    };

    enum Button {

        /* Binary buttons */
        ATTACK          = 0,
        USE             = 1,
        JUMP            = 2,
        CROUCH          = 3,
        TURN180         = 4,
        ALTATTACK       = 5,
        RELOAD          = 6,
        ZOOM            = 7,

        SPEED           = 8,
        STRAFE          = 9,

        MOVE_RIGHT      = 10,
        MOVE_LEFT       = 11,
        MOVE_BACKWARD   = 12,
        MOVE_FORWARD    = 13,
        TURN_RIGHT      = 14,
        TURN_LEFT       = 15,
        LOOK_UP         = 16,
        LOOK_DOWN       = 17,
        MOVE_UP         = 18,
        MOVE_DOWN       = 19,
        LAND            = 20,

        SELECT_WEAPON1  = 21,
        SELECT_WEAPON2  = 22,
        SELECT_WEAPON3  = 23,
        SELECT_WEAPON4  = 24,
        SELECT_WEAPON5  = 25,
        SELECT_WEAPON6  = 26,
        SELECT_WEAPON7  = 27,
        SELECT_WEAPON8  = 28,
        SELECT_WEAPON9  = 29,
        SELECT_WEAPON0  = 30,

        SELECT_NEXT_WEAPON          = 31,
        SELECT_PREV_WEAPON          = 32,
        DROP_SELECTED_WEAPON        = 33,

        ACTIVATE_SELECTED_ITEM      = 34,
        SELECT_NEXT_ITEM            = 35,
        SELECT_PREV_ITEM            = 36,
        DROP_SELECTED_ITEM          = 37,

        /* Delta buttons */
        LOOK_UP_DOWN_DELTA          = 38,
        TURN_LEFT_RIGHT_DELTA       = 39,
        MOVE_FORWARD_BACKWARD_DELTA = 40,
        MOVE_LEFT_RIGHT_DELTA       = 41,
        MOVE_UP_DOWN_DELTA          = 42,
    };

}
#endif
