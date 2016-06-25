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

#include <string.h>

#include "dobject.h"
#include "dobjtype.h"
#include "doomtype.h"
#include "name.h"
#include "d_player.h"
//#include "namedef.h"
//#include "sc_man.h"
//#include "sc_man_tokens.h"

#define VIZ_GV_USER_COUNT           30
#define VIZ_GV_SLOTS_SIZE           10
#define VIZ_MAX_PLAYERS             MAXPLAYERS // 8
#define VIZ_MAX_PLAYER_NAME_LEN     MAXPLAYERNAME + 1 //(15 + 1 = 16)

struct VIZGameState{
    unsigned int VERSION;
    char VERSION_STR[8];
    size_t SM_SIZE;

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

    bool PLAYER_READY_TO_RESPAWN;
    unsigned int PLAYER_NUMBER;

    // OTHER PLAYERS
    unsigned int PLAYER_COUNT;
    bool PLAYERS_IN_GAME[VIZ_MAX_PLAYERS];
    char PLAYERS_NAME[VIZ_MAX_PLAYERS][VIZ_MAX_PLAYER_NAME_LEN];
    int PLAYERS_FRAGCOUNT[VIZ_MAX_PLAYERS];
};

int VIZ_CheckItem(FName name);

int VIZ_CheckItem(PClass *type);

const char* VIZ_CheckItemType(PClass *type);

bool VIZ_CheckSelectedWeaponState();

int VIZ_CheckSelectedWeapon();

int VIZ_CheckWeaponAmmo(AWeapon* weapon);

int VIZ_CheckSelectedWeaponAmmo();

int VIZ_CheckSlotAmmo(unsigned int slot);

int VIZ_CheckSlotWeapons(unsigned int slot);

void VIZ_GameStateInit();

void VIZ_GameStateTic();

void VIZ_GameStateClose();

#endif
