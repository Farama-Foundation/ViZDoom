#ifndef __VIZIA_SHARED_MEMORY_H__
#define __VIZIA_SHARED_MEMORY_H__

#include "interprocess/shared_memory_object.hpp"
#include "interprocess/mapped_region.hpp"

using namespace boost::interprocess;

extern shared_memory_object *viziaSM;

#define VBT_ATTACK 0
#define VBT_USE 1
#define VBT_JUMP 2
#define VBT_CROUCH 3
//#define VBT_TURN180 4
#define VBT_ALTATTACK 5
#define VBT_RELOAD 6
#define VBT_ZOOM 7

#define VBT_SPEED 8
#define VBT_STRAFE 9

#define VBT_MOVERIGHT 10
#define VBT_MOVELEFT 11
#define VBT_BACK 12
#define VBT_FORWARD 13
#define VBT_RIGHT 14
#define VBT_LEFT 15
#define VBT_LOOKUP 16
#define VBT_LOOKDOWN 17
#define VBT_MOVEUP 18
#define VBT_MOVEDOWN 19
//#define VBT_SHOWSCORES 20

#define VBT_WEAPON1 21
#define VBT_WEAPON2 22
#define VBT_WEAPON3 23
#define VBT_WEAPON4 24
#define VBT_WEAPON5 25
#define VBT_WEAPON6 26
#define VBT_WEAPON7 27

#define VBT_WEAPONNEXT 28
#define VBT_WEAPONPREV 29

#define VBT_SIZE 30

struct ViziaInputSMStruct{
    int MS_X;
    int MS_Y;
    bool BT[VBT_SIZE];
};

struct ViziaGameVarsSMStruct{
    int TIC;

    int SCREEN_WIDTH;
    int SCREEN_HEIGHT;

    bool MAP_FINISHED;

    bool PLAYER_DEAD;

    int PLAYER_KILLCOUNT;
    int PLAYER_ITEMCOUNT;
    int PLAYER_SECRETCOUNT;
    int PLAYER_FRAGCOUNT; //for multiplayer

    bool PLAYER_ONGROUND;

    int PLAYER_HEALTH;
    int PLAYER_ARMOR;

    int PLAYER_EQUIPPED_WEAPON;
    int PLAYER_EQUIPPED_WEAPON_AMMO;

    int PLAYER_AMMO[4];
    bool PLAYER_WEAPON[7];
    bool PLAYER_KEY[3];
};

#define VIZIA_SCREEN_SM_NAME "ViziaScreen"
#define VIZIA_GAME_VARS_SM_NAME "ViziaGameVars"
#define VIZIA_INPUT_SM_NAME "ViziaInput"

#define VIZIA_SM_NAME "ViziaSM"

void Vizia_SMInit();

int Vizia_SMSetSize(int scr_w, int src_h);

size_t Vizia_SMGetInputRegionBeginning();
size_t Vizia_SMGetGameVarsRegionBeginning();
size_t Vizia_SMGetScreenRegionBeginning();

void Vizia_SMClose();

#endif
