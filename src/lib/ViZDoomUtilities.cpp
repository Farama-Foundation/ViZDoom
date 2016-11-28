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

#include "ViZDoomUtilities.h"

namespace vizdoom {

    double doomTicsToMs(double tics, unsigned int ticrate) {
        return 1000.0 / ticrate * tics;
    }

    double msToDoomTics(double ms, unsigned int ticrate) {
        return static_cast<double>(ticrate) / 1000.0 * ms;
    }

    double doomTicsToSec(double tics, unsigned int ticrate) {
        return 1.0 / ticrate * tics;
    }

    double secToDoomTics(double sec, unsigned int ticrate) {
        return static_cast<double>(ticrate) * sec;
    }


    double doomFixedToDouble(int doomFixed) {
        return static_cast<double>(doomFixed) / 65536.0;
    }

    double doomFixedToDouble(double doomFixed) {
        return doomFixed / 65536.0;
    }

    #define CASE_ENUM(e) case e: return #e;

    std::string modeToString(Mode mode){
        switch(mode) {
            CASE_ENUM(PLAYER)
            CASE_ENUM(SPECTATOR)
            CASE_ENUM(ASYNC_PLAYER)
            CASE_ENUM(ASYNC_SPECTATOR)
            default: return "UNKNOWN";
        }
    }

    std::string screenFormatToString(ScreenFormat screenFormat){
        switch(screenFormat) {
            CASE_ENUM(CRCGCB)
            CASE_ENUM(RGB24)
            CASE_ENUM(RGBA32)
            CASE_ENUM(ARGB32)
            CASE_ENUM(CBCGCR)
            CASE_ENUM(BGR24)
            CASE_ENUM(BGRA32)
            CASE_ENUM(ABGR32)
            CASE_ENUM(GRAY8)
            CASE_ENUM(DOOM_256_COLORS8)
            default: return "UNKNOWN";
        }
    }

    std::string automapModeToString(AutomapMode automapMode){
        switch(automapMode) {
            CASE_ENUM(NORMAL)
            CASE_ENUM(WHOLE)
            CASE_ENUM(OBJECTS)
            CASE_ENUM(OBJECTS_WITH_SIZE)
            default: return "UNKNOWN";
        }
    }

    std::string gameVariableToString(GameVariable gameVariable){
        switch(gameVariable) {
            CASE_ENUM(KILLCOUNT)
            CASE_ENUM(ITEMCOUNT)
            CASE_ENUM(SECRETCOUNT)
            CASE_ENUM(FRAGCOUNT)
            CASE_ENUM(DEATHCOUNT)
            CASE_ENUM(HEALTH)
            CASE_ENUM(ARMOR)
            CASE_ENUM(DEAD)
            CASE_ENUM(ON_GROUND)
            CASE_ENUM(ATTACK_READY)
            CASE_ENUM(ALTATTACK_READY)
            CASE_ENUM(SELECTED_WEAPON)
            CASE_ENUM(SELECTED_WEAPON_AMMO)
            CASE_ENUM(AMMO0)
            CASE_ENUM(AMMO1)
            CASE_ENUM(AMMO2)
            CASE_ENUM(AMMO3)
            CASE_ENUM(AMMO4)
            CASE_ENUM(AMMO5)
            CASE_ENUM(AMMO6)
            CASE_ENUM(AMMO7)
            CASE_ENUM(AMMO8)
            CASE_ENUM(AMMO9)
            CASE_ENUM(WEAPON0)
            CASE_ENUM(WEAPON1)
            CASE_ENUM(WEAPON2)
            CASE_ENUM(WEAPON3)
            CASE_ENUM(WEAPON4)
            CASE_ENUM(WEAPON5)
            CASE_ENUM(WEAPON6)
            CASE_ENUM(WEAPON7)
            CASE_ENUM(WEAPON8)
            CASE_ENUM(WEAPON9)
            CASE_ENUM(POSITION_X)
            CASE_ENUM(POSITION_Y)
            CASE_ENUM(POSITION_Z)
            CASE_ENUM(PLAYER_NUMBER)
            CASE_ENUM(PLAYER_COUNT)
            CASE_ENUM(PLAYER1_FRAGCOUNT)
            CASE_ENUM(PLAYER2_FRAGCOUNT)
            CASE_ENUM(PLAYER3_FRAGCOUNT)
            CASE_ENUM(PLAYER4_FRAGCOUNT)
            CASE_ENUM(PLAYER5_FRAGCOUNT)
            CASE_ENUM(PLAYER6_FRAGCOUNT)
            CASE_ENUM(PLAYER7_FRAGCOUNT)
            CASE_ENUM(PLAYER8_FRAGCOUNT)
            CASE_ENUM(USER1)
            CASE_ENUM(USER2)
            CASE_ENUM(USER3)
            CASE_ENUM(USER4)
            CASE_ENUM(USER5)
            CASE_ENUM(USER6)
            CASE_ENUM(USER7)
            CASE_ENUM(USER8)
            CASE_ENUM(USER9)
            CASE_ENUM(USER10)
            CASE_ENUM(USER11)
            CASE_ENUM(USER12)
            CASE_ENUM(USER13)
            CASE_ENUM(USER14)
            CASE_ENUM(USER15)
            CASE_ENUM(USER16)
            CASE_ENUM(USER17)
            CASE_ENUM(USER18)
            CASE_ENUM(USER19)
            CASE_ENUM(USER20)
            CASE_ENUM(USER21)
            CASE_ENUM(USER22)
            CASE_ENUM(USER23)
            CASE_ENUM(USER24)
            CASE_ENUM(USER25)
            CASE_ENUM(USER26)
            CASE_ENUM(USER27)
            CASE_ENUM(USER28)
            CASE_ENUM(USER29)
            CASE_ENUM(USER30)
            CASE_ENUM(USER31)
            CASE_ENUM(USER32)
            CASE_ENUM(USER33)
            CASE_ENUM(USER34)
            CASE_ENUM(USER35)
            CASE_ENUM(USER36)
            CASE_ENUM(USER37)
            CASE_ENUM(USER38)
            CASE_ENUM(USER39)
            CASE_ENUM(USER40)
            CASE_ENUM(USER41)
            CASE_ENUM(USER42)
            CASE_ENUM(USER43)
            CASE_ENUM(USER44)
            CASE_ENUM(USER45)
            CASE_ENUM(USER46)
            CASE_ENUM(USER47)
            CASE_ENUM(USER48)
            CASE_ENUM(USER49)
            CASE_ENUM(USER50)
            CASE_ENUM(USER51)
            CASE_ENUM(USER52)
            CASE_ENUM(USER53)
            CASE_ENUM(USER54)
            CASE_ENUM(USER55)
            CASE_ENUM(USER56)
            CASE_ENUM(USER57)
            CASE_ENUM(USER58)
            CASE_ENUM(USER59)
            CASE_ENUM(USER60)
            default: return "UNKNOWN";
        }
    }

    std::string buttonToString(Button button){
        switch(button) {
            CASE_ENUM(ATTACK)
            CASE_ENUM(USE)
            CASE_ENUM(JUMP)
            CASE_ENUM(CROUCH)
            CASE_ENUM(TURN180)
            CASE_ENUM(ALTATTACK)
            CASE_ENUM(RELOAD)
            CASE_ENUM(ZOOM)
            CASE_ENUM(SPEED)
            CASE_ENUM(STRAFE)
            CASE_ENUM(MOVE_RIGHT)
            CASE_ENUM(MOVE_LEFT)
            CASE_ENUM(MOVE_BACKWARD)
            CASE_ENUM(MOVE_FORWARD)
            CASE_ENUM(TURN_RIGHT)
            CASE_ENUM(TURN_LEFT)
            CASE_ENUM(LOOK_UP)
            CASE_ENUM(LOOK_DOWN)
            CASE_ENUM(MOVE_UP)
            CASE_ENUM(MOVE_DOWN)
            CASE_ENUM(LAND)
            CASE_ENUM(SELECT_WEAPON1)
            CASE_ENUM(SELECT_WEAPON2)
            CASE_ENUM(SELECT_WEAPON3)
            CASE_ENUM(SELECT_WEAPON4)
            CASE_ENUM(SELECT_WEAPON5)
            CASE_ENUM(SELECT_WEAPON6)
            CASE_ENUM(SELECT_WEAPON7)
            CASE_ENUM(SELECT_WEAPON8)
            CASE_ENUM(SELECT_WEAPON9)
            CASE_ENUM(SELECT_WEAPON0)
            CASE_ENUM(SELECT_NEXT_WEAPON)
            CASE_ENUM(SELECT_PREV_WEAPON)
            CASE_ENUM(DROP_SELECTED_WEAPON)
            CASE_ENUM(ACTIVATE_SELECTED_ITEM)
            CASE_ENUM(SELECT_NEXT_ITEM)
            CASE_ENUM(SELECT_PREV_ITEM)
            CASE_ENUM(DROP_SELECTED_ITEM)
            CASE_ENUM(LOOK_UP_DOWN_DELTA)
            CASE_ENUM(TURN_LEFT_RIGHT_DELTA)
            CASE_ENUM(MOVE_FORWARD_BACKWARD_DELTA)
            CASE_ENUM(MOVE_LEFT_RIGHT_DELTA)
            CASE_ENUM(MOVE_UP_DOWN_DELTA)
            default: return "UNKNOWN";
        }
    }

    bool isBinaryButton(Button button){
        return button < BINARY_BUTTON_COUNT;
    }

    bool isDeltaButton(Button button) {
        return button >= BINARY_BUTTON_COUNT && button < BUTTON_COUNT;
    }
}