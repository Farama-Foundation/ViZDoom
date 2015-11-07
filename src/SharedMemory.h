#ifndef SHEREDMEMORY_H
#define SHEREDMEMORY_H

#include <boost/interprocess/managed_shared_memory.hpp>

using namespace boost::interprocess;

struct ViziaInputSMStruct{
    int MS_DELTAX;
    int MS_DELTAY;

    bool BT_ATTACK;
    bool BT_USE;
    bool BT_JUMP;
    bool BT_CROUCH;

    bool BT_SPEED;
    bool BT_STRAFE;

    bool BT_MOVERIGHT;
    bool BT_MOVELEFT;
    bool BT_BACK;
    bool BT_FORWARD;
    bool BT_RIGHT;
    bool BT_LEFT;
    bool BT_MOVEUP;
    bool BT_MOVEDOWN;

    int BT_WEAPON;
};

struct ViziaGameDataSMStruct{
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

shared_memory_object *viziaScreenSM;
shared_memory_object *viziaGameDataSM;
shared_memory_object *viziaInputSM;

const char* const viziaScreenSMName = "ViziaScreen";
const char* const viziaGameDataSMName = "ViziaGameData";
const char* const viziaInputSMName = "ViziaInput";

BYTE *viziaScreen;
struct ViziaInputSMStruct *viziaInput;
struct ViziaGameDataSMStruct *viziaGameData;

#endif //VIZIA_SHERED_MEMORY_H
