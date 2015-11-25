#ifndef __VIZIA_SHARED_MEMORY_H__
#define __VIZIA_SHARED_MEMORY_H__

#include <boost/interprocess/shared_memory_object.hpp>
#include <boost/interprocess/mapped_region.hpp>

namespace bip = boost::interprocess;

extern bip::shared_memory_object viziaSM;

#define A_ATTACK 0
#define A_USE 1
#define A_JUMP 2
#define A_CROUCH 3
#define A_TURN180 4
#define A_ALTATTACK 5
#define A_RELOAD 6
#define A_ZOOM 7

#define A_SPEED 8
#define A_STRAFE 9

#define A_MOVERIGHT 10
#define A_MOVELEFT 11
#define A_BACK 12
#define A_FORWARD 13
#define A_RIGHT 14
#define A_LEFT 15
#define A_LOOKUP 16
#define A_LOOKDOWN 17
#define A_MOVEUP 18
#define A_MOVEDOWN 19
//#define A_SHOWSCORES 20

#define A_WEAPON1 21
#define A_WEAPON2 22
#define A_WEAPON3 23
#define A_WEAPON4 24
#define A_WEAPON5 25
#define A_WEAPON6 26
#define A_WEAPON7 27

#define A_WEAPONNEXT 28
#define A_WEAPONPREV 29

#define A_BT_SIZE 30

struct ViziaInputStruct{
    int MS_X;
    int MS_Y;
    int MS_MAX_X;
    int MS_MAX_Y;
    bool BT[A_BT_SIZE];
    bool BT_AVAILABLE[A_BT_SIZE];
};

struct ViziaGameVarsStruct{
    unsigned int GAME_TIC;

    unsigned int SCREEN_WIDTH;
    unsigned int SCREEN_HEIGHT;
    size_t SCREEN_PITCH;
    size_t SCREEN_SIZE;
    int SCREEN_FORMAT;

    int MAP_REWARD;

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
