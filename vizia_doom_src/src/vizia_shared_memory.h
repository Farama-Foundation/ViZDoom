#ifndef __VIZIA_SHARED_MEMORY_H__
#define __VIZIA_SHARED_MEMORY_H__

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace bip = boost::interprocess;

extern bip::shared_memory_object viziaSM;

#define V_ATTACK 0
#define V_USE 1
#define V_JUMP 2
#define V_CROUCH 3
#define V_TURN180 4
#define V_ALTATTACK 5
#define V_RELOAD 6
#define V_ZOOM 7

#define V_SPEED 8
#define V_STRAFE 9

#define V_MOVERIGHT 10
#define V_MOVELEFT 11
#define V_BACK 12
#define V_FORWARD 13
#define V_RIGHT 14
#define V_LEFT 15
#define V_LOOKUP 16
#define V_LOOKDOWN 17
#define V_MOVEUP 18
#define V_MOVEDOWN 19
//#define V_SHOWSCORES 20

#define V_WEAPON1 21
#define V_WEAPON2 22
#define V_WEAPON3 23
#define V_WEAPON4 24
#define V_WEAPON5 25
#define V_WEAPON6 26
#define V_WEAPON7 27

#define V_WEAPONNEXT 28
#define V_WEAPONPREV 29

#define V_BT_SIZE 30

struct ViziaInputStruct{
    int MS_X;
    int MS_Y;
    bool BT[V_BT_SIZE];
};

struct ViziaGameVarsStruct{
    int GAME_TIC;

    int SCREEN_WIDTH;
    int SCREEN_HEIGHT;

    int MAP_START_TIC;
    int MAP_TIC;

    int MAP_KILLCOUNT;
    int MAP_ITEMCOUNT;
    int MAP_SECRETCOUNT;
    bool MAP_END;

    bool PLAYER_DEAD;

    int PLAYER_KILLCOUNT;
    int PLAYER_ITEMCOUNT;
    int PLAYER_SECRETCOUNT;
    int PLAYER_FRAGCOUNT; //for multiplayer

    bool PLAYER_ONGROUND;

    int PLAYER_HEALTH;
    int PLAYER_ARMOR;

    int PLAYER_SELECTED_WEAPON;
    int PLAYER_SELECTED_WEAPON_AMMO;

    int PLAYER_AMMO[4];
    bool PLAYER_WEAPON[7];
    bool PLAYER_KEY[3];
};

#define VIZIA_SM_NAME_BASE "ViziaSM"

void Vizia_SMInit(const char * id);

void Vizia_SMSetSize(int screenWidth, int screenHeight);

size_t Vizia_SMGetInputRegionBeginning();
size_t Vizia_SMGetGameVarsRegionBeginning();
size_t Vizia_SMGetScreenRegionBeginning();

void Vizia_SMClose();

#endif
