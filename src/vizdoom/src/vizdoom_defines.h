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

#ifndef __VIZDOOM_DEFINES_H__
#define __VIZDOOM_DEFINES_H__

#include <stdlib.h>

extern bool vizdoomNextTic;
extern bool vizdoomUpdate;
extern unsigned int vizdoomLastUpdate;

#define VIZDOOM_VERSION 102
#define VIZDOOM_VERSION_STR "1.0.2"

#define VIZDOOM_TIME (level.starttime + level.maptime)

#define VIZDOOM_BT_ATTACK         0
#define VIZDOOM_BT_USE            1
#define VIZDOOM_BT_JUMP           2
#define VIZDOOM_BT_CROUCH         3
#define VIZDOOM_BT_TURN180        4
#define VIZDOOM_BT_ALTATTACK      5
#define VIZDOOM_BT_RELOAD         6
#define VIZDOOM_BT_ZOOM           7

#define VIZDOOM_BT_SPEED          8
#define VIZDOOM_BT_STRAFE         9

#define VIZDOOM_BT_MOVE_RIGHT     10
#define VIZDOOM_BT_MOVE_LEFT      11
#define VIZDOOM_BT_MOVE_BACK      12
#define VIZDOOM_BT_MOVE_FORWARD   13
#define VIZDOOM_BT_TURN_RIGHT     14
#define VIZDOOM_BT_TURN_LEFT      15
#define VIZDOOM_BT_LOOK_UP        16
#define VIZDOOM_BT_LOOK_DOWN      17
#define VIZDOOM_BT_MOVE_UP        18
#define VIZDOOM_BT_MOVE_DOWN      19
#define VIZDOOM_BT_LAND 20
//#define VIZDOOM_BT_SHOWSCORES 20

#define VIZDOOM_BT_SELECT_WEAPON1 21
#define VIZDOOM_BT_SELECT_WEAPON2 22
#define VIZDOOM_BT_SELECT_WEAPON3 23
#define VIZDOOM_BT_SELECT_WEAPON4 24
#define VIZDOOM_BT_SELECT_WEAPON5 25
#define VIZDOOM_BT_SELECT_WEAPON6 26
#define VIZDOOM_BT_SELECT_WEAPON7 27
#define VIZDOOM_BT_SELECT_WEAPON8 28
#define VIZDOOM_BT_SELECT_WEAPON9 29
#define VIZDOOM_BT_SELECT_WEAPON0 30

#define VIZDOOM_BT_SELECT_NEXT_WEAPON 31
#define VIZDOOM_BT_SELECT_PREV_WEAPON 32
#define VIZDOOM_BT_DROP_SELECTED_WEAPON 33

#define VIZDOOM_BT_ACTIVATE_SELECTED_ITEM 34
#define VIZDOOM_BT_SELECT_NEXT_ITEM 35
#define VIZDOOM_BT_SELECT_PREV_ITEM 36
#define VIZDOOM_BT_DROP_SELECTED_ITEM 37

#define VIZDOOM_BT_VIEW_PITCH 38
#define VIZDOOM_BT_VIEW_ANGLE 39
#define VIZDOOM_BT_FORWARD_BACKWARD 40
#define VIZDOOM_BT_LEFT_RIGHT 41
#define VIZDOOM_BT_UP_DOWN 42

#define VIZDOOM_BT_CMD_BT_SIZE 38
#define VIZDOOM_BT_AXIS_BT_SIZE 5
#define VIZDOOM_BT_SIZE 43

#define VIZDOOM_GV_USER_SIZE 30

#define VIZDOOM_GV_SLOTS_SIZE 10

#define VIZDOOM_MAX_PLAYERS 8 //MAXPLAYERS //(8)
#define VIZDOOM_MAX_PLAYER_NAME_LEN 16 //MAXPLAYERNAME+1 //(15 + 1 = 16)

struct ViZDoomInputStruct{
    int BT[VIZDOOM_BT_SIZE];
    bool BT_AVAILABLE[VIZDOOM_BT_SIZE];
    int BT_MAX_VALUE[VIZDOOM_BT_AXIS_BT_SIZE];
};

struct ViZDoomGameVarsStruct{

    unsigned int VERSION;
    char VERSION_STR[8];

    unsigned int GAME_TIC;
    int GAME_STATE;
    int GAME_ACTION;
    unsigned int GAME_SEED;
    unsigned int GAME_STATIC_SEED;
    bool GAME_SETTINGS_CONTROLLER;
    bool NET_GAME;

    unsigned int SCREEN_WIDTH;
    unsigned int SCREEN_HEIGHT;
    size_t SCREEN_PITCH;
    size_t SCREEN_SIZE;
    int SCREEN_FORMAT;

    unsigned int MAP_START_TIC;
    unsigned int MAP_TIC;

    int MAP_REWARD;

    int MAP_USER_VARS[VIZDOOM_GV_USER_SIZE];

    int MAP_KILLCOUNT;
    int MAP_ITEMCOUNT;
    int MAP_SECRETCOUNT;
    bool MAP_END;

    bool PLAYER_HAS_ACTOR;
    bool PLAYER_DEAD;

    char PLAYER_NAME[VIZDOOM_MAX_PLAYER_NAME_LEN];
    int PLAYER_KILLCOUNT;
    int PLAYER_ITEMCOUNT;
    int PLAYER_SECRETCOUNT;
    int PLAYER_FRAGCOUNT;
    int PLAYER_DEATHCOUNT;

    bool PLAYER_ON_GROUND;

    int PLAYER_HEALTH;
    int PLAYER_ARMOR;

    bool PLAYER_ATTACK_READY;
    bool PLAYER_ALTATTACK_READY;

    int PLAYER_SELECTED_WEAPON;
    int PLAYER_SELECTED_WEAPON_AMMO;

    int PLAYER_AMMO[VIZDOOM_GV_SLOTS_SIZE];
    int PLAYER_WEAPON[VIZDOOM_GV_SLOTS_SIZE];

    int PLAYERS_COUNT;
    char PLAYERS_NAME[VIZDOOM_MAX_PLAYERS][VIZDOOM_MAX_PLAYER_NAME_LEN];
    int PLAYERS_FRAGCOUNT[VIZDOOM_MAX_PLAYERS];

};

#define VIZDOOM_SCREEN_CRCGCB 0
#define VIZDOOM_SCREEN_CRCGCBDB 1
#define VIZDOOM_SCREEN_RGB24 2
#define VIZDOOM_SCREEN_RGBA32 3
#define VIZDOOM_SCREEN_ARGB32 4
#define VIZDOOM_SCREEN_CBCGCR 5
#define VIZDOOM_SCREEN_CBCGCRDB 6
#define VIZDOOM_SCREEN_BGR24 7
#define VIZDOOM_SCREEN_BGRA32 8
#define VIZDOOM_SCREEN_ABGR32 9
#define VIZDOOM_SCREEN_GRAY8 10
#define VIZDOOM_SCREEN_DEPTH_BUFFER8 11
#define VIZDOOM_SCREEN_DOOM_256_COLORS8 12

#endif
