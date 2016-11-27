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

#ifndef __VIZ_GAME_H__
#define __VIZ_GAME_H__

#include <viz_defines.h>
#include <viz_labels.h>
#include <viz_shared_memory.h>
#include <string.h>

#include "dobject.h"
#include "dobjtype.h"
#include "doomtype.h"
#include "name.h"
#include "d_player.h"
//#include "namedef.h"
//#include "sc_man.h"
//#include "sc_man_tokens.h"

#define VIZ_GV_USER_COUNT 60
#define VIZ_GV_SLOTS_SIZE 10

struct VIZGameState{
    // VERSION
    unsigned int VERSION;
    char VERSION_STR[8];

    // SM
    size_t SM_SIZE;
    size_t SM_REGION_OFFSET[VIZ_SM_REGION_COUNT];
    size_t SM_REGION_SIZE[VIZ_SM_REGION_COUNT];
    bool SM_REGION_WRITEABLE[VIZ_SM_REGION_COUNT];

    // GAME
    unsigned int GAME_TIC;
    int GAME_STATE;
    int GAME_ACTION;
    unsigned int GAME_STATIC_SEED;
    bool GAME_SETTINGS_CONTROLLER;
    bool GAME_NETGAME;
    bool GAME_MULTIPLAYER;
    bool DEMO_RECORDING;
    bool DEMO_PLAYBACK;

    // SCREEN
    unsigned int SCREEN_WIDTH;
    unsigned int SCREEN_HEIGHT;
    size_t SCREEN_PITCH;
    size_t SCREEN_SIZE;
    int SCREEN_FORMAT;

    bool DEPTH_BUFFER;
    bool LABELS;
    bool AUTOMAP;

    // MAP
    unsigned int MAP_START_TIC;
    unsigned int MAP_TIC;

    int MAP_REWARD;
    int MAP_USER_VARS[VIZ_GV_USER_COUNT];

    int MAP_KILLCOUNT;
    int MAP_ITEMCOUNT;
    int MAP_SECRETCOUNT;
    bool MAP_END;

    // PLAYER
    bool PLAYER_HAS_ACTOR;
    bool PLAYER_DEAD;

    char PLAYER_NAME[VIZ_MAX_PLAYER_NAME_LEN];
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

    int PLAYER_AMMO[VIZ_GV_SLOTS_SIZE];
    int PLAYER_WEAPON[VIZ_GV_SLOTS_SIZE];

    double PLAYER_POSITION[3];

    bool PLAYER_READY_TO_RESPAWN;
    unsigned int PLAYER_NUMBER;

    // OTHER PLAYERS
    unsigned int PLAYER_COUNT;
    bool PLAYER_N_IN_GAME[VIZ_MAX_PLAYERS];
    char PLAYER_N_NAME[VIZ_MAX_PLAYERS][VIZ_MAX_PLAYER_NAME_LEN];
    int PLAYER_N_FRAGCOUNT[VIZ_MAX_PLAYERS];

    //LABELS
    unsigned int LABEL_COUNT;
    VIZLabel LABEL[VIZ_MAX_LABELS];

};

void VIZ_GameStateInit();

void VIZ_GameStateTic();

void VIZ_GameStateUpdate();

void VIZ_GameStateUpdateLabels();

void VIZ_GameStateInitNew();

void VIZ_GameStateClose();

#endif
