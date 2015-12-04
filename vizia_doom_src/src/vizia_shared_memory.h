#ifndef __VIZIA_SHARED_MEMORY_H__
#define __VIZIA_SHARED_MEMORY_H__

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace bip = boost::interprocess;

extern bip::shared_memory_object viziaSM;

#define VIZIA_BT_ATTACK 0
#define VIZIA_BT_USE 1
#define VIZIA_BT_JUMP 2
#define VIZIA_BT_CROUCH 3
#define VIZIA_BT_TURN180 4
#define VIZIA_BT_ALTATTACK 5
#define VIZIA_BT_RELOAD 6
#define VIZIA_BT_ZOOM 7

#define VIZIA_BT_SPEED 8
#define VIZIA_BT_STRAFE 9

#define VIZIA_BT_MOVERIGHT 10
#define VIZIA_BT_MOVELEFT 11
#define VIZIA_BT_BACK 12
#define VIZIA_BT_FORWARD 13
#define VIZIA_BT_RIGHT 14
#define VIZIA_BT_LEFT 15
#define VIZIA_BT_LOOKUP 16
#define VIZIA_BT_LOOKDOWN 17
#define VIZIA_BT_MOVEUP 18
#define VIZIA_BT_MOVEDOWN 19
//#define VIZIA_BT_SHOWSCORES 20

#define VIZIA_BT_WEAPON1 21
#define VIZIA_BT_WEAPON2 22
#define VIZIA_BT_WEAPON3 23
#define VIZIA_BT_WEAPON4 24
#define VIZIA_BT_WEAPON5 25
#define VIZIA_BT_WEAPON6 26
#define VIZIA_BT_WEAPON7 27

#define VIZIA_BT_WEAPONNEXT 28
#define VIZIA_BT_WEAPONPREV 29

#define VIZIA_BT_SIZE 30

#define VIZIA_GV_USER_SIZE 30

struct ViziaInputStruct{
    int MS_X;
    int MS_Y;
    int MS_MAX_X;
    int MS_MAX_Y;
    bool BT[VIZIA_BT_SIZE];
    bool BT_AVAILABLE[VIZIA_BT_SIZE];
};

struct ViziaGameVarsStruct{
    unsigned int GAME_TIC;
    unsigned int GAME_SEED;
    unsigned int GAME_STATIC_SEED;

    unsigned int SCREEN_WIDTH;
    unsigned int SCREEN_HEIGHT;
    size_t SCREEN_PITCH;
    size_t SCREEN_SIZE;
    int SCREEN_FORMAT;

    int MAP_REWARD;
    int MAP_SHAPING_REWARD;

    int MAP_USER_VARS[VIZIA_GV_USER_SIZE];

    unsigned int MAP_START_TIC;
    unsigned int MAP_TIC;

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

    int PLAYER_AMMO[10];
    bool PLAYER_WEAPON[10];
    bool PLAYER_KEY[10];
};

#define VIZIA_SM_NAME_BASE "ViziaSM"

void Vizia_SMInit(const char * id);

size_t Vizia_SMGetInputRegionBeginning();
size_t Vizia_SMGetGameVarsRegionBeginning();
size_t Vizia_SMGetScreenRegionBeginning();

void Vizia_SMClose();

#endif
