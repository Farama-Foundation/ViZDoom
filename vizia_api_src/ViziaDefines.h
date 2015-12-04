#ifndef __VIZIA_DEFINES_H__
#define __VIZIA_DEFINES_H__

namespace Vizia{

    enum ScreenFormat {
        CRCGCB = 0,
        CRCGCBCA = 1,
        RGB24 = 2,
        RGBA32 = 3,
        ARGB32 = 4,
        CBCGCR = 5,
        CBCGCRCA = 6,
        BGR24 = 7,
        BGRA32 = 8,
        ABGR32 = 9
    };

    enum GameVar {
        KILLCOUNT = 0,
        ITEMCOUNT = 1,
        SECRETCOUNT = 2,
        HEALTH = 3,
        ARMOR = 4,
        SELECTED_WEAPON = 5,
        SELECTED_WEAPON_AMMO = 6,
        AMMO1 = 7,
        AMMO2 = 8,
        AMMO3 = 9,
        AMMO4 = 10,
        WEAPON1 = 11,
        WEAPON2 = 12,
        WEAPON3 = 13,
        WEAPON4 = 14,
        WEAPON5 = 15,
        WEAPON6 = 16,
        WEAPON7 = 17,
        KEY1 = 18,
        KEY2 = 19,
        KEY3 = 20,
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
        UNDEFINED_VAR
    };

    enum Button {
        ATTACK = 0,
        USE = 1,
        JUMP = 2,
        CROUCH = 3,
        TURN180 = 4,
        ALTATTACK = 5,
        RELOAD = 6,
        ZOOM = 7,

        SPEED = 8,
        STRAFE = 9,

        MOVERIGHT = 10,
        MOVELEFT = 11,
        BACK = 12,
        FORWARD = 13,
        RIGHT = 14,
        LEFT = 15,
        LOOKUP = 16,
        LOOKDOWN = 17,
        MOVEUP = 18,
        MOVEDOWN = 19,
        //SHOWSCORES 20

        SELECT_WEAPON1 = 21,
        SELECT_WEAPON2 = 22,
        SELECT_WEAPON3 = 23,
        SELECT_WEAPON4 = 24,
        SELECT_WEAPON5 = 25,
        SELECT_WEAPON6 = 26,
        SELECT_WEAPON7 = 27,

        SELECT_NEXT_WEAPON = 28,
        SELECT_PREV_WEAPON = 29,
        UNDEFINED_BUTTON
    };

    static const int ButtonsNumber = 30;


    static const int DOOM_AMMO_CLIP = 0;
    static const int DOOM_AMMO_SHELL = 1;
    static const int DOOM_AMMO_ROCKET = 2;
    static const int DOOM_AMMO_CELL = 3;

    static const int DOOM_WEAPON_FIST = 0;
    static const int DOOM_WEAPON_CHAINSAW = 0;
    static const int DOOM_WEAPON_PISTOL = 1;
    static const int DOOM_WEAPON_SHOTGUN = 3;
    static const int DOOM_WEAPON_SSG = 3;
    static const int DOOM_WEAPON_SUPER_SHOTGUN = 3;
    static const int DOOM_WEAPON_CHAINGUN = 4;
    static const int DOOM_WEAPON_ROCKET_LUNCHER = 5;
    static const int DOOM_WEAPON_PLASMA_GUN = 6;
    static const int DOOM_WEAPON_BFG = 7;

    static const int DOOM_KEY_BLUE = 0;
    static const int DOOM_KEY_RED = 1;
    static const int DOOM_KEY_YELLOW = 2;

}
#endif